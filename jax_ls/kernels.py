"""The truncated free-space Green's kernel, and its application by FFT.

Implements the method of Vico, Greengard and Ferrando (2016), arXiv:1604.03155:
the free-space kernel of the Helmholtz operator is truncated to a disc of radius
``L``, which makes its Fourier transform an entire function and turns the volume
potential into an *aperiodic* convolution that a zero-padded FFT computes
exactly.

`tests/test_vico_greengard.py` reproduces section 5.1 of that paper and holds
the volume potential to a relative L2 error of 1.7e-15 against the closed-form
solution, so it is the regression guard for anything in this module.
"""

import numpy as np
import scipy as sp

import jax
import jax.numpy as jnp

from flax import struct

from .domain import Domain

Array = jax.Array

# Relative tolerance used to detect the removable singularity |s| == omega of
# the truncated kernel.  See `fourier_green_function`.
SINGULARITY_RTOL = 1e-8

# Default zero-padding factor used for the aperiodic FFT convolution.
DEFAULT_PAD = 4

# Default truncation radius, as a multiple of the diameter of the domain.
DEFAULT_TRUNCATION_FACTOR = 1.05


def fourier_green_function(L: float, k: float, s: Array) -> Array:
  """Evaluate the truncated Green's function in the Fourier domain.

  This is the Fourier transform of the free-space Green's function of the
  Helmholtz operator truncated to a disc of radius ``L``,

    Ghat_L(s) = N(s) / (s^2 - k^2),

    N(s) = 1 + (i pi / 2) L s H1_0(L k) J_1(L s)
             - (i pi / 2) L k H1_1(L k) J_0(L s),

  where ``H1_p`` is the Hankel function of the first kind and ``J_p`` the Bessel
  function of the first kind.

  ``N`` vanishes at ``|s| = k`` (a consequence of the Wronskian identity
  ``J_0(z) Y_1(z) - J_1(z) Y_0(z) = -2 / (pi z)``), so the quotient has a
  *removable* singularity there.  Evaluating the quotient directly computes
  ``0 / 0``, which is not a rare edge case: the mode set of the padded grid
  contains ``|s| = k`` exactly whenever ``omega`` is chosen as a fixed number of
  points per wavelength.  Near ``|s| = k`` the quotient also loses accuracy to
  cancellation.  Both are avoided by switching to the limit

    lim_{|s| -> k} Ghat_L(s) = (i pi L^2 / 4) [ H1_0(Lk) J_0(Lk)
                                              + H1_1(Lk) J_1(Lk) ]

  whenever ``|s^2 - k^2| <= SINGULARITY_RTOL * k^2``.

  Evaluated host-side in numpy/scipy, since the Hankel functions have no JAX
  equivalent; this is why kernel construction is not jittable.

  Args:
    L: The truncation radius.
    k: The wavenumber (omega).
    s: The Fourier frequency variable.

  Returns:
    The transformed Green's function evaluated at s, as a numpy array.
  """
  s = np.asarray(s, dtype=float)
  k = float(k)

  hankel_0, hankel_1 = (
    sp.special.hankel1(0, L * k),
    sp.special.hankel1(1, L * k),
  )

  numerator = (
    1
    + (1j * np.pi / 2 * L * s) * hankel_0 * sp.special.jv(1, L * s)
    - (1j * np.pi / 2 * L * k) * hankel_1 * sp.special.jv(0, L * s)
  )
  denominator = np.square(s) - np.square(k)

  # Value of the quotient at the removable singularity, by L'Hopital's rule.
  limit = (1j * np.pi * L**2 / 4) * (
    hankel_0 * sp.special.jv(0, L * k) + hankel_1 * sp.special.jv(1, L * k)
  )

  singular = np.abs(denominator) <= SINGULARITY_RTOL * np.square(k)
  # The unused branch of `where` is still evaluated, so keep it finite.
  safe_denominator = np.where(singular, 1.0, denominator)

  return np.where(singular, limit, numerator / safe_denominator)


@struct.dataclass
class GreenKernel:
  """The truncated Green's kernel of one frequency, ready to apply by FFT.

  Carries the `Domain` it was built for.  That is free: a `Domain` has no array
  leaves, so the only leaf here is `ghat`.  It also means `apply` knows the
  shape to crop back to without being told.

  The frequency is deliberately *not* stored.  It is baked into `ghat`, and
  keeping it out means a wide-band sweep can vary the frequency without
  changing the pytree structure and forcing a recompile.

  Attributes:
    domain: The grid this kernel was built for.
    ghat: The truncated kernel in the Fourier domain, of shape
      ``(pad * m, pad * n)``.
    truncation_radius: The radius L the kernel was truncated at.
    pad: Zero-padding factor of the aperiodic convolution.
  """

  domain: Domain
  ghat: Array
  truncation_radius: float = struct.field(pytree_node=False)
  pad: int = struct.field(pytree_node=False, default=DEFAULT_PAD)

  @classmethod
  def create(
    cls,
    domain: Domain,
    omega: float,
    pad: int = DEFAULT_PAD,
    truncation_radius: float | None = None,
  ) -> "GreenKernel":
    """Build the truncated kernel for a domain and a frequency.

    Args:
      domain: The grid to build the kernel for.
      omega: Angular frequency.
      pad: Zero-padding factor used for the aperiodic FFT convolution.
      truncation_radius: Radius L at which the Green's function is truncated.
        Defaults to DEFAULT_TRUNCATION_FACTOR times the diameter of the domain.

    Returns:
      An initialized GreenKernel.

    Raises:
      ValueError: If the padding is degenerate, or if the truncation radius and
        the padding are incompatible with the size of the domain.
    """
    if pad < 2:
      raise ValueError(f"the padding factor must be at least 2, got {pad}")

    L = (
      DEFAULT_TRUNCATION_FACTOR * domain.diameter
      if truncation_radius is None
      else float(truncation_radius)
    )

    # The truncated kernel must agree with the free-space kernel on every pair
    # of points of the domain.
    if L < domain.diameter:
      raise ValueError(
        f"truncation radius {L:.4g} is smaller than the diameter of the domain "
        f"{domain.diameter:.4g}; the truncated kernel would not represent the "
        "volume potential"
      )
    # The periodic convolution on the padded box must not alias: the nearest
    # periodic image has to fall outside the support of the truncated kernel.
    for length, label in ((domain.ax, "ax"), (domain.ay, "ay")):
      if pad * length < length + L:
        raise ValueError(
          f"padding {pad} is too small along {label}: the padded period "
          f"{pad * length:.4g} must exceed {label} + L = {length + L:.4g}. "
          "Increase `pad` or decrease `truncation_radius`."
        )

    # Centred integer mode indices, -N/2 ... N/2 - 1, matching `ifftshift`.
    kx = jnp.arange(pad * domain.n) - (pad * domain.n) // 2
    ky = jnp.arange(pad * domain.m) - (pad * domain.m) // 2

    Kx, Ky = jnp.meshgrid(
      (2 * np.pi / (pad * domain.ax)) * kx,
      (2 * np.pi / (pad * domain.ay)) * ky,
    )
    S = jnp.sqrt(jnp.square(Kx) + jnp.square(Ky))

    ghat = jnp.fft.ifftshift(jnp.asarray(fourier_green_function(L, omega, S)))

    return cls(domain=domain, ghat=ghat, truncation_radius=L, pad=pad)

  def _convolve(self, u: Array, ghat: Array) -> Array:
    """Zero-pad, multiply in the Fourier domain, and crop back.

    The zero padding is what makes the periodic FFT convolution reproduce the
    aperiodic volume potential.
    """
    rows, cols = self.domain.shape

    padded = jnp.zeros(ghat.shape, dtype=ghat.dtype)
    padded = padded.at[:rows, :cols].set(jnp.reshape(u, (rows, cols)))

    transformed = jnp.fft.ifft2(ghat * jnp.fft.fft2(padded))

    return jnp.reshape(transformed[:rows, :cols], (-1,))

  def apply(self, u: Array) -> Array:
    """Apply the volume potential ``G u``."""
    return self._convolve(u, self.ghat)

  def apply_adjoint(self, u: Array) -> Array:
    """Apply the conjugate volume potential ``G^H u``."""
    return self._convolve(u, jnp.conj(self.ghat))
