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

import jax
import jax.numpy as jnp
from jax import custom_jvp, jit

from flax import struct

from .domain import Domain
from .solvers import GmresConfig, SolveInfo, solve, solve_with_info
from .kernels import (
  DEFAULT_PAD,
  DEFAULT_TRUNCATION_FACTOR,
  SINGULARITY_RTOL,
  GreenKernel,
  fourier_green_function,
)

Array = jax.Array

__all__ = [
  "DEFAULT_PAD",
  "DEFAULT_TRUNCATION_FACTOR",
  "SINGULARITY_RTOL",
  "Domain",
  "GreenKernel",
  "LippSchwinParams",
  "apply_conj_green_function",
  "apply_green_function",
  "apply_lipp_schwin",
  "apply_lipp_schwin_adj",
  "fourier_green_function",
  "GmresConfig",
  "SolveInfo",
  "init_params",
  "ls_solver",
  "ls_solver_adj",
  "ls_solver_with_info",
  "ls_solver_adj_with_info",
  "ls_solver_batched",
  "ls_solver_batched_adj",
  "ls_solver_batched_sigma",
  "ls_solver_field",
  "ls_solver_u",
]


@struct.dataclass
class LippSchwinParams:
  """Parameters for the Lippmann-Schwinger solver at one frequency.

  A thin composition of a `Domain` (pure geometry, no array leaves) and a
  `GreenKernel` (the frequency-dependent part).  The grid coordinates and
  spacings are exposed as properties that read through to the domain, so there
  is exactly one definition of the grid convention -- see `jax_ls.domain`.

  Attributes:
    kernel: The truncated Green's kernel, which carries the domain.
    omega: The angular frequency of the wavefield.
  """

  kernel: GreenKernel
  omega: float

  # -- geometry, read through to the domain --------------------------------
  @property
  def domain(self) -> Domain:
    """The computational grid."""
    return self.kernel.domain

  @property
  def X(self) -> Array:
    """The x-coordinates of the computational grid, of shape (m, n)."""
    return self.domain.X

  @property
  def Y(self) -> Array:
    """The y-coordinates of the computational grid, of shape (m, n)."""
    return self.domain.Y

  @property
  def hx(self) -> float:
    """Grid spacing in x, equal to ax / n."""
    return self.domain.hx

  @property
  def hy(self) -> float:
    """Grid spacing in y, equal to ay / m."""
    return self.domain.hy

  @property
  def shape(self) -> tuple[int, int]:
    """Shape (m, n) of the computational grid."""
    return self.domain.shape

  @property
  def n_dof(self) -> int:
    """Number of degrees of freedom on the grid."""
    return self.domain.n_dof

  @property
  def cell_area(self) -> float:
    """Quadrature weight of one grid cell."""
    return self.domain.cell_area

  # -- kernel --------------------------------------------------------------
  @property
  def GFFT(self) -> Array:
    """The truncated Green's function in the Fourier domain."""
    return self.kernel.ghat


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

  Convenience wrapper over `Domain` and `GreenKernel.create`; equivalent to

    LippSchwinParams(GreenKernel.create(Domain(ax, ay, n, m), omega), omega)

  The grid follows the periodic convention described in `jax_ls.domain`: ``n``
  points of spacing ``ax / n`` covering ``[-ax/2, ax/2)``, and likewise in y.
  Rectangular domains are supported: the padded periods are ``pad * ax`` and
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
  domain = Domain(ax=ax, ay=ay, n=n, m=m)
  kernel = GreenKernel.create(
    domain, omega, pad=pad, truncation_radius=truncation_radius
  )
  return LippSchwinParams(kernel=kernel, omega=omega)


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
  kernel = params.kernel
  return kernel.apply_adjoint(u) if conjugate else kernel.apply(u)


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


def _lipp_schwin_operator(params: LippSchwinParams, nu_vect: Array):
  """The Lippmann-Schwinger operator as a callable, for the linear solvers."""
  return lambda x: apply_lipp_schwin(params, nu_vect, x)


def _lipp_schwin_operator_adj(params: LippSchwinParams, nu_vect: Array):
  """The adjoint-like operator as a callable, for the linear solvers."""
  return lambda x: apply_lipp_schwin_adj(params, nu_vect, x)


@jit
def ls_solver(
  params: LippSchwinParams,
  nu_vect: Array,
  f: Array,
  config: GmresConfig = GmresConfig(),
  x0: Array | None = None,
) -> Array:
  """Solve the Lippmann-Schwinger equation for the density.

  Args:
    params: Configuration parameters for the solver.
    nu_vect: Contrast/perturbation vector.
    f: Right-hand side vector.
    config: GMRES settings.  Defaults are unchanged from before.
    x0: Optional initial guess.

  Returns:
    The solution density sigma.
  """
  return solve(_lipp_schwin_operator(params, nu_vect), f, config, x0)


@jit
def ls_solver_adj(
  params: LippSchwinParams,
  nu_vect: Array,
  f: Array,
  config: GmresConfig = GmresConfig(),
  x0: Array | None = None,
) -> Array:
  """Solve the adjoint Lippmann-Schwinger equation for the density.

  Args:
    params: Configuration parameters for the solver.
    nu_vect: Contrast/perturbation vector.
    f: Right-hand side vector.
    config: GMRES settings.
    x0: Optional initial guess.

  Returns:
    The adjoint solution density.
  """
  return solve(_lipp_schwin_operator_adj(params, nu_vect), f, config, x0)


@jit
def ls_solver_with_info(
  params: LippSchwinParams,
  nu_vect: Array,
  f: Array,
  config: GmresConfig = GmresConfig(),
  x0: Array | None = None,
) -> tuple[Array, SolveInfo]:
  """Solve, and report whether GMRES converged.

  GMRES's own ``info`` is always 0 and carries no information, so the residual
  is measured here at the cost of one extra operator application.  Worth it
  when the contrast is large: see `jax_ls.solvers`.

  Returns:
    A tuple of (density, `SolveInfo`).
  """
  return solve_with_info(_lipp_schwin_operator(params, nu_vect), f, config, x0)


@jit
def ls_solver_adj_with_info(
  params: LippSchwinParams,
  nu_vect: Array,
  f: Array,
  config: GmresConfig = GmresConfig(),
  x0: Array | None = None,
) -> tuple[Array, SolveInfo]:
  """Solve the adjoint equation, and report whether GMRES converged.

  Returns:
    A tuple of (adjoint density, `SolveInfo`).
  """
  return solve_with_info(
    _lipp_schwin_operator_adj(params, nu_vect), f, config, x0
  )


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
