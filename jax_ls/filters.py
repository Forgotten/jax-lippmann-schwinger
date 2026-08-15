"""Band-limiting the unknown contrast.

At frequency ``omega`` the Born kernel is ``exp(i omega (s_hat - r_hat) . x)`` and
``|omega (s_hat - r_hat)| <= 2 omega``, so **no experiment at that frequency
carries information about spatial frequencies of the contrast above** ``2
omega``.  That is the half-wavelength diffraction limit: features finer than
``lambda / 2`` are invisible.

Restricting the unknown to that band is what makes each linearised subproblem of
a recursive-linearization sweep well conditioned: the number of unknowns
``O(omega^2)`` then tracks the number of independent measurements
``N_src * N_recv = O(omega^2)``, and ``J^H J`` stops being singular on the modes
the data cannot see.  Without it an optimiser puts energy into unrecoverable high
modes, which the next frequency then has to undo.

This band-limiting is the *only* regularisation used by

  C. Borges, A. Gillman and L. Greengard, "High resolution inverse scattering in
  two dimensions using recursive linearization", SIAM J. Imaging Sci. 10(2),
  641-664 (2017); arXiv:1608.06871

who truncate a sine series at ``m1 + m2 <= S(k) = floor(2k)`` -- the same
``|xi| <= 2 omega`` limit expressed in their basis -- and apply no Tikhonov term
at all.  It is offered as an *option* here rather than imposed: pass
``band_factor=None`` to the inversion routines to disable it, in which case a
nonzero damping should be supplied instead, since the normal equations are then
singular on the invisible modes.

On the grid of a `Domain` the DFT modes are ``xi_j = 2 pi j / ax``, so the band
``|xi| <= factor * omega`` keeps mode indices ``|j| <= factor * omega * ax /
(2 pi)``.  At ``omega = 8 pi``, ``ax = 1``, ``n = 64`` that retains 197 of 4096
modes -- a twentyfold reduction -- against the ``pi * (omega ax / pi)^2 = 201``
the area of the disc predicts.
"""

import jax
import jax.numpy as jnp

from .domain import Domain

Array = jax.Array

# The Born band limit: |xi| <= 2 omega.  Exposed as a default rather than
# hardcoded so that a deliberately conservative sweep can ask for less.
DEFAULT_BAND_FACTOR = 2.0


def _mode_radius(domain: Domain) -> Array:
  """``|xi|`` at every DFT mode of the grid, in FFT (unshifted) order."""
  # `fftfreq(n, d=1/n)` gives the signed integer mode indices in FFT order.
  xi_x = 2 * jnp.pi * jnp.fft.fftfreq(domain.n, d=1.0 / domain.n) / domain.ax
  xi_y = 2 * jnp.pi * jnp.fft.fftfreq(domain.m, d=1.0 / domain.m) / domain.ay
  return jnp.sqrt(jnp.square(xi_x)[None, :] + jnp.square(xi_y)[:, None])


def band_mask(
  domain: Domain, omega: float, factor: float = DEFAULT_BAND_FACTOR
) -> Array:
  """The retained modes, as a boolean array of shape ``domain.shape``.

  Args:
    domain: The grid.
    omega: Angular frequency.
    factor: Multiple of `omega` to keep; 2 is the physical Born limit.

  Returns:
    True where ``|xi| <= factor * omega``, in FFT order.
  """
  return _mode_radius(domain) <= factor * omega


def band_limit(
  domain: Domain,
  omega: float,
  nu_vect: Array,
  factor: float = DEFAULT_BAND_FACTOR,
) -> Array:
  """Project a contrast onto the modes an experiment at `omega` can see.

  This is an orthogonal projector: idempotent and self-adjoint, so it can be
  composed into a least-squares operator pair without breaking the adjoint
  relation the solver relies on.

  The result is real, as the contrast is: the retained mode set is symmetric
  under ``xi -> -xi``, so the filtered field has no spurious imaginary part.

  Args:
    domain: The grid.
    omega: Angular frequency.
    nu_vect: Flattened contrast.
    factor: Multiple of `omega` to keep.

  Returns:
    The filtered contrast, flattened.
  """
  mask = band_mask(domain, omega, factor)
  spectrum = jnp.fft.fft2(domain.grid(nu_vect))
  return domain.ravel(jnp.real(jnp.fft.ifft2(jnp.where(mask, spectrum, 0.0))))


def band_index(
  domain: Domain, omega: float, factor: float = DEFAULT_BAND_FACTOR
) -> float:
  """Largest retained mode index along x, ``factor * omega * ax / (2 pi)``.

  The sampling rules of a wide-band sweep are naturally written in terms of this
  number: it is the ``S(k)`` of Borges, Gillman and Greengard.
  """
  return factor * omega * domain.ax / (2 * jnp.pi)


def band_dimension(
  domain: Domain, omega: float, factor: float = DEFAULT_BAND_FACTOR
) -> int:
  """How many degrees of freedom survive the projection.

  The dimension of the search space a band-limited inversion actually explores,
  as opposed to `domain.n_dof`.
  """
  return int(jnp.sum(band_mask(domain, omega, factor)))
