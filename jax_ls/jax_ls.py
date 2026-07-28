"""Lippmann-Schwinger solver for the 2-D Helmholtz equation.

The scattered field ``u_s`` produced by a compactly supported contrast ``nu``
probed by an incident field ``u_i`` satisfies

  Delta u_s + omega^2 (1 + nu) u_s = -omega^2 nu u_i

together with the Sommerfeld radiation condition.  Writing the induced source
as ``sigma = omega^2 nu (u_i + u_s)`` and letting ``G`` be the volume potential
of the outgoing fundamental solution (``Delta G + omega^2 G = -delta``), this
becomes the Lippmann-Schwinger equation

  (-I + omega^2 nu G) sigma = -omega^2 nu u_i,      u_s = G sigma.

``G`` is applied in quasi-linear time by FFT using the truncated kernel of
Vico, Greengard and Ferrando (2016).

Grid convention
---------------
The domain ``[-ax/2, ax/2) x [-ay/2, ay/2)`` is treated as *periodic*: it is
discretised with ``n`` (resp. ``m``) points of spacing ``hx = ax / n`` (resp.
``hy = ay / m``), with no duplicated endpoint.  This is exactly the grid the
FFT-based convolution assumes -- the padded transform has period ``pad * ax``
on ``pad * n`` points -- so the quadrature weight ``hx * hy`` is consistent
with the operator.
"""

from functools import partial

import numpy as np
import scipy as sp

import jax
import jax.numpy as jnp
from jax import custom_jvp, jit

from flax import struct

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
class LippSchwinParams:
  """Parameters for the Lippmann-Schwinger solver.

  Attributes:
    GFFT: The truncated Green's function in the Fourier domain, of shape
      (pad * m, pad * n).
    omega: The angular frequency of the wavefield.
    X: The x-coordinates of the computational grid, of shape (m, n).
    Y: The y-coordinates of the computational grid, of shape (m, n).
    hx: Grid spacing in x, equal to ax / n.
    hy: Grid spacing in y, equal to ay / m.
  """

  # Truncated Green's function in Fourier domain.
  GFFT: Array
  # Frequency.
  omega: float
  # Grid in x.
  X: Array
  # Grid in y.
  Y: Array
  # Grid spacings.  Static: they are geometry, not data to be differentiated.
  hx: float = struct.field(pytree_node=False)
  hy: float = struct.field(pytree_node=False)

  @property
  def shape(self) -> tuple[int, int]:
    """Shape (m, n) of the computational grid."""
    return self.X.shape

  @property
  def n_dof(self) -> int:
    """Number of degrees of freedom on the grid."""
    return self.X.size

  @property
  def cell_area(self) -> float:
    """Quadrature weight of one grid cell."""
    return self.hx * self.hy


def init_params(
  ax: float,
  ay: float,
  n: int,
  m: int,
  omega: float,
  pad: int = DEFAULT_PAD,
  truncation_radius: float | None = None,
) -> LippSchwinParams:
  """Initialize the parameters for the Lippmann-Schwinger solver.

  The grid follows the periodic convention described in the module docstring:
  ``n`` points of spacing ``ax / n`` covering ``[-ax/2, ax/2)``, and likewise in
  y.  Rectangular domains are supported: the padded periods are ``pad * ax`` and
  ``pad * ay`` independently.

  Args:
    ax: Length of the domain in the x direction.
    ay: Length of the domain in the y direction.
    n: Number of discretization points in the x direction.
    m: Number of discretization points in the y direction.
    omega: Angular frequency.
    pad: Zero-padding factor used for the aperiodic FFT convolution.
    truncation_radius: Radius L at which the Green's function is truncated.
      Defaults to DEFAULT_TRUNCATION_FACTOR times the diameter of the domain.

  Returns:
    A LippSchwinParams object containing the initialized parameters.

  Raises:
    ValueError: If the grid is degenerate, or if the truncation radius and the
      padding are incompatible with the size of the domain.
  """
  if n < 2 or m < 2:
    raise ValueError(
      f"the grid must have at least 2 points per axis, got {(n, m)}"
    )
  if pad < 2:
    raise ValueError(f"the padding factor must be at least 2, got {pad}")

  hx, hy = ax / n, ay / m

  # Periodic grid: no duplicated endpoint, spacing exactly ax / n.
  x = -ax / 2 + hx * jnp.arange(n)
  y = -ay / 2 + hy * jnp.arange(m)
  X, Y = jnp.meshgrid(x, y)

  diameter = float(np.hypot(ax, ay))
  L = (
    DEFAULT_TRUNCATION_FACTOR * diameter
    if truncation_radius is None
    else float(truncation_radius)
  )

  # The truncated kernel must agree with the free-space kernel on every pair of
  # points of the domain.
  if L < diameter:
    raise ValueError(
      f"truncation radius {L:.4g} is smaller than the diameter of the domain "
      f"{diameter:.4g}; the truncated kernel would not represent the volume "
      "potential"
    )
  # The periodic convolution on the padded box must not alias: the nearest
  # periodic image has to fall outside the support of the truncated kernel.
  for length, label in ((ax, "ax"), (ay, "ay")):
    if pad * length < length + L:
      raise ValueError(
        f"padding {pad} is too small along {label}: the padded period "
        f"{pad * length:.4g} must exceed {label} + L = {length + L:.4g}. "
        "Increase `pad` or decrease `truncation_radius`."
      )

  # Centred integer mode indices, -N/2 ... N/2 - 1, matching `ifftshift` below.
  kx = jnp.arange(pad * n) - (pad * n) // 2
  ky = jnp.arange(pad * m) - (pad * m) // 2

  Kx, Ky = jnp.meshgrid(
    (2 * np.pi / (pad * ax)) * kx, (2 * np.pi / (pad * ay)) * ky
  )
  S = jnp.sqrt(jnp.square(Kx) + jnp.square(Ky))

  GFFT = jnp.fft.ifftshift(jnp.asarray(fourier_green_function(L, omega, S)))

  return LippSchwinParams(GFFT=GFFT, omega=omega, X=X, Y=Y, hx=hx, hy=hy)


def _apply_kernel(
  params: LippSchwinParams, u: Array, conjugate: bool = False
) -> Array:
  """Convolve u with the truncated Green's function by FFT.

  The input is zero-padded into the corner of the (pad * m, pad * n) transform
  grid, multiplied by the kernel in the Fourier domain, and the physical block
  is extracted again.  The zero padding is what makes the periodic FFT
  convolution reproduce the aperiodic volume potential.

  Args:
    params: Configuration parameters for the solver.
    u: Input wavefield as a flattened array.
    conjugate: If True, convolve with the conjugate kernel instead.

  Returns:
    The convolved wavefield as a flattened array.
  """
  rows, cols = params.shape
  ghat = jnp.conj(params.GFFT) if conjugate else params.GFFT

  padded = jnp.zeros(ghat.shape, dtype=ghat.dtype)
  padded = padded.at[:rows, :cols].set(jnp.reshape(u, (rows, cols)))

  transformed = jnp.fft.ifft2(ghat * jnp.fft.fft2(padded))

  return jnp.reshape(transformed[:rows, :cols], (-1,))


@jit
def apply_green_function(params: LippSchwinParams, u: Array) -> Array:
  """Apply the Green's function via FFT-based convolution.

  Args:
    params: Configuration parameters for the solver.
    u: Input wavefield as a flattened array.

  Returns:
    The convolved wavefield G u as a flattened array.
  """
  return _apply_kernel(params, u)


@jit
def apply_conj_green_function(params: LippSchwinParams, u: Array) -> Array:
  """Apply the conjugate of the Green's function via FFT.

  This is used when computing adjoints.

  Args:
    params: Configuration parameters for the solver.
    u: Input wavefield.

  Returns:
    The result of G^H u.
  """
  return _apply_kernel(params, u, conjugate=True)


@jit
def apply_lipp_schwin(
  params: LippSchwinParams, nu_vect: Array, u: Array
) -> Array:
  """Apply the Lippmann-Schwinger operator to a wavefield.

  Args:
    params: Configuration parameters for the solver.
    nu_vect: Contrast/perturbation vector.
    u: Input wavefield.

  Returns:
    The result of (-I + omega^2 * nu * G) u.
  """
  return -u + jnp.square(params.omega) * nu_vect * _apply_kernel(params, u)


@jit
def apply_lipp_schwin_adj(
  params: LippSchwinParams, nu_vect: Array, u: Array
) -> Array:
  """Apply the adjoint-like Lippmann-Schwinger operator.

  This returns ``(-I + omega^2 * nu * G^H) u``, which is the similarity
  transform ``D A^H D^{-1}`` of the true adjoint, with ``D = diag(nu)``.  It is
  the operator that appears when the adjoint problem is written in the same
  "incident field plus scattered field" form as the forward problem, which is
  how `near_field` assembles it.

  Args:
    params: Configuration parameters for the solver.
    nu_vect: Contrast/perturbation vector.
    u: Input wavefield.

  Returns:
    The result of (-I + omega^2 * nu * G^H) u.
  """
  return -u + jnp.square(params.omega) * nu_vect * _apply_kernel(
    params, u, conjugate=True
  )


@jit
def ls_solver(params: LippSchwinParams, nu_vect: Array, f: Array) -> Array:
  """Solve the Lippmann-Schwinger equation for the density.

  Note that `jax.scipy.sparse.linalg.gmres` already defaults to
  ``solve_method="batched"``; that option selects how the inner least-squares
  problem is solved and has nothing to do with multiple right-hand sides.
  Several right-hand sides are handled by `vmap`-ing this function.

  Args:
    params: Configuration parameters for the solver.
    nu_vect: Contrast/perturbation vector.
    f: Right-hand side vector.

  Returns:
    The solution density sigma.
  """
  sigma, _ = jax.scipy.sparse.linalg.gmres(
    lambda x: apply_lipp_schwin(params, nu_vect, x), f
  )
  return sigma


@jit
def ls_solver_adj(params: LippSchwinParams, nu_vect: Array, f: Array) -> Array:
  """Solve the adjoint Lippmann-Schwinger equation for the density.

  Args:
    params: Configuration parameters for the solver.
    nu_vect: Contrast/perturbation vector.
    f: Right-hand side vector.

  Returns:
    The adjoint solution density.
  """
  sigma, _ = jax.scipy.sparse.linalg.gmres(
    lambda x: apply_lipp_schwin_adj(params, nu_vect, x), f
  )
  return sigma


@partial(custom_jvp, nondiff_argnums=(0, 2))
def ls_solver_field(
  params: LippSchwinParams, nu_vect: Array, f: Array
) -> Array:
  """Solve for the wavefield u = G sigma directly.

  Differentiable in forward mode with respect to `nu_vect`; the right-hand side
  `f` is held fixed.

  Args:
    params: Configuration parameters for the solver.
    nu_vect: Contrast/perturbation vector.
    f: Right-hand side vector.

  Returns:
    The solution wavefield u.
  """
  return apply_green_function(params, ls_solver(params, nu_vect, f))


@ls_solver_field.defjvp
def ls_solver_field_jvp(
  params: LippSchwinParams,
  f: Array,
  primals: tuple[Array],
  tangents: tuple[Array],
) -> tuple[Array, Array]:
  """Forward-mode rule for `ls_solver_field`.

  Differentiating ``A(nu) sigma = f`` at fixed ``f`` gives

    A delta_sigma = -omega^2 delta_nu * u,      u = G sigma,

  so the tangent is obtained by one extra solve with the same operator.  Note
  that ``u`` here is the *scattered* field, because ``f`` does not depend on
  ``nu``.  When the right-hand side does depend on ``nu`` -- as it does for the
  near-field map -- the total field appears instead; see
  `near_field.near_field_map_forward_mode`.

  Args:
    params: Configuration parameters for the solver.
    f: Right-hand side vector.
    primals: Primal values (nu_vect,).
    tangents: Tangent values (delta_nu_vect,).

  Returns:
    A tuple containing the primal solution and the tangent solution.
  """
  (nu_vect,) = primals
  (delta_nu_vect,) = tangents

  u = ls_solver_field(params, nu_vect, f)
  rhs = -(params.omega**2) * u * delta_nu_vect
  delta_u = ls_solver_field(params, nu_vect, rhs)

  return u, delta_u


# ---------------------------------------------------------------------------
# Deprecated aliases.
#
# `ls_solver_batched` and `ls_solver_batched_sigma` were separate copies of
# `ls_solver`: `solve_method="batched"` is already the GMRES default, and
# batching over right-hand sides is done with `vmap`.
# ---------------------------------------------------------------------------
ls_solver_batched = ls_solver
ls_solver_batched_sigma = ls_solver
ls_solver_batched_adj = ls_solver_adj
ls_solver_u = ls_solver_field
