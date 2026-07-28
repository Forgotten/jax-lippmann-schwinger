"""Near-field map built on the Lippmann-Schwinger solver, and its derivatives.

The near-field map sends a contrast ``nu`` to the scattered field measured on an
observation manifold, for each of a family of point sources placed on the same
manifold:

  F[nu]_{s,r} = u_s^{(s)}(x_r),        x_s, x_r on the observation circle.

Which function to use for what
------------------------------
JAX allows a function to carry *either* a custom JVP *or* a custom VJP, not
both, and the linearised solver cannot be transposed automatically (GMRES is not
a transposable primitive).  There are therefore three distinct entry points, and
picking the wrong one silently produces a wrong derivative:

  ``near_field_map``
      The map itself, matrix valued, with no AD rule.  Use it to generate data
      and to plot.

  ``near_field_map_forward_mode``
      The map with a ``custom_jvp`` whose rule is the Born approximation.  Use
      it with ``jax.jvp`` / ``jax.linearize``.

  ``near_field_map_reverse_mode``
      The map with a ``custom_vjp`` implementing the *transpose* ``J^T``, which
      is the convention JAX's reverse mode expects.  Use it with ``jax.grad`` /
      ``jax.vjp``.

  ``born_linearization``
      Explicit application of ``J``.

  ``born_adjoint``
      Explicit application of the *Hermitian* adjoint ``J^H``, satisfying
      ``<w, J t> = <J^H w, t>``.  This is what a matrix-free least-squares
      solver (CGLS/LSQR, i.e. Gauss-Newton) needs.  It is deliberately *not* an
      autodiff entry point: ``J^H`` differs from ``J^T`` by conjugation, and
      feeding it to ``jax.grad`` gives a wrong gradient.

Sign convention
---------------
``green_function`` is the fundamental solution, ``Delta G + omega^2 G = -delta``.
``NearFieldParams.G_sample`` stores ``-G`` sampled between the grid and the
observation manifold, which is why the sampling contractions below carry an
explicit minus sign.
"""

from functools import partial

import scipy as sp

import jax
import jax.numpy as jnp
from jax import custom_jvp, custom_vjp, jit, vmap

from flax import struct

from .jax_ls import LippSchwinParams, init_params
from .jax_ls import apply_conj_green_function, apply_green_function
from .jax_ls import ls_solver, ls_solver_adj

Array = jax.Array


def green_function(r: Array, omega: float) -> Array:
  """Evaluate the fundamental solution of the Helmholtz equation.

  Note the sign: it satisfies Delta G(x,y) + omega^2 G(x,y) = -delta(x,y).

  Args:
    r: Distance between points.
    omega: Angular frequency.

  Returns:
    The value of the Green's function at distance r.
  """
  return (1j / 4) * sp.special.hankel1(0, omega * r)


@struct.dataclass
class NearFieldParams:
  """Configuration for the near-field map and its derivatives.

  Attributes:
    ls_params: Lippmann-Schwinger configuration.
    G_sample: Minus the Green's function sampled between the grid and the
      observation manifold, of shape (n_dof, n_theta).  It plays two roles: its
      columns are the incident fields of the point sources, and it is the kernel
      that samples the scattered field at the receivers.
  """

  # Lippmann-Schwinger parameters (grid, kernel, frequency, spacings).
  ls_params: LippSchwinParams
  # Green's function in space.
  G_sample: Array

  # The grid spacings live on `ls_params`; exposing them here keeps the call
  # sites short without storing a second, independent copy.
  @property
  def hx(self) -> float:
    """Grid spacing in x."""
    return self.ls_params.hx

  @property
  def hy(self) -> float:
    """Grid spacing in y."""
    return self.ls_params.hy

  @property
  def cell_area(self) -> float:
    """Quadrature weight of one grid cell."""
    return self.ls_params.cell_area

  @property
  def n_theta(self) -> int:
    """Number of sources, which is also the number of receivers."""
    return self.G_sample.shape[1]


def init_params_near_field(
  ax: float,
  ay: float,
  n: int,
  m: int,
  r: float,
  n_theta: int,
  omega: float,
  **kwargs,
) -> NearFieldParams:
  """Initialize parameters for the near-field measurement system.

  Args:
    ax: Length of the domain in the x direction.
    ay: Length of the domain in the y direction.
    n: Number of discretization points in the x direction.
    m: Number of discretization points in the y direction.
    r: Radius of the observation manifold.
    n_theta: Number of samples/sources on the observation manifold.
    omega: Angular frequency.
    **kwargs: Forwarded to `init_params` (e.g. `pad`, `truncation_radius`).

  Returns:
    An initialized NearFieldParams object.
  """
  params = init_params(ax, ay, n, m, omega, **kwargs)

  d_theta = jnp.pi * 2 / n_theta
  theta = jnp.linspace(jnp.pi, 3 * jnp.pi - d_theta, n_theta)

  # Defining the observation (and sampling) manifold.
  X_s = r * jnp.concatenate(
    [
      jnp.cos(theta).reshape((n_theta, 1)),
      jnp.sin(theta).reshape((n_theta, 1)),
    ],
    axis=1,
  )

  # Computing the distances to each evaluation point.
  R = jnp.sqrt(
    jnp.square(params.X.reshape((-1, 1)) - X_s[:, 0].reshape((1, -1)))
    + jnp.square(params.Y.reshape((-1, 1)) - X_s[:, 1].reshape((1, -1)))
  )

  return NearFieldParams(params, -green_function(R, omega))


# ---------------------------------------------------------------------------
# Building blocks.  Every public function below is assembled from these, so the
# forward map, its linearisation and its adjoint cannot drift apart.
# ---------------------------------------------------------------------------
def _incident_rhs(params: NearFieldParams, nu_vect: Array) -> Array:
  """Right-hand side of the integral equation, one column per source."""
  return (
    -(params.ls_params.omega**2) * nu_vect.reshape((-1, 1)) * params.G_sample
  )


def _solve_all_sources(
  params: NearFieldParams, nu_vect: Array, rhs: Array
) -> Array:
  """Solve the integral equation for every column of `rhs`."""
  solve = vmap(
    partial(ls_solver, params.ls_params, nu_vect), in_axes=1, out_axes=1
  )
  return solve(rhs)


def _solve_all_sources_adj(
  params: NearFieldParams, nu_vect: Array, rhs: Array
) -> Array:
  """Solve the adjoint integral equation for every column of `rhs`."""
  solve = vmap(
    partial(ls_solver_adj, params.ls_params, nu_vect), in_axes=1, out_axes=1
  )
  return solve(rhs)


def _apply_green_all(params: NearFieldParams, Sigma: Array) -> Array:
  """Apply the Green's function to every column of `Sigma`."""
  green = vmap(
    partial(apply_green_function, params.ls_params), in_axes=1, out_axes=1
  )
  return green(Sigma)


def _apply_conj_green_all(params: NearFieldParams, Sigma: Array) -> Array:
  """Apply the conjugate Green's function to every column of `Sigma`."""
  green = vmap(
    partial(apply_conj_green_function, params.ls_params), in_axes=1, out_axes=1
  )
  return green(Sigma)


def _total_field(params: NearFieldParams, Sigma: Array) -> Array:
  """Total field (incident plus scattered) for every source."""
  return _apply_green_all(params, Sigma) + params.G_sample


def _sample_near_field(params: NearFieldParams, Sigma: Array) -> Array:
  """Sample the scattered field on the observation manifold.

  The result is indexed (source, receiver).  The minus sign is the
  ``G_sample = -G`` convention described in the module docstring.
  """
  return -(Sigma.T @ params.G_sample) * params.cell_area


def _adjoint_apply(
  params: NearFieldParams, nu_vect: Array, Sigma: Array, cotangent: Array
) -> Array:
  """Apply the Hermitian adjoint J^H to a cotangent matrix.

  The adjoint problem is assembled in exactly the same "incident plus scattered"
  form as the forward problem, with the conjugate kernel:

    u_i^H  = omega^2 conj(G_sample) w         (field radiated back by the residual)
    rhs^H  = -omega^2 nu * u_i^H
    sigma^H = A_adj^{-1} rhs^H
    u^H    = G^H sigma^H + u_i^H
    J^H w  = -cell_area * sum_sources conj(u_total) * u^H

  The contraction of `cotangent` with `G_sample` pairs the *source* index of the
  cotangent with the receiver index of the kernel.  That is legitimate because
  the linearised map is reciprocal -- ``J t`` is a symmetric matrix for every
  ``t`` -- so ``J`` annihilates the antisymmetric part of any cotangent and both
  pairings define the same adjoint on the range of ``J``.

  Args:
    params: Configuration for the near-field map.
    nu_vect: Contrast vector.
    Sigma: Forward densities, one column per source.
    cotangent: Cotangent, of shape (n_theta, n_theta).

  Returns:
    The cotangent with respect to `nu_vect`, of shape (n_dof,).
  """
  omega = params.ls_params.omega

  u_total = _total_field(params, Sigma)

  # Incident field of the adjoint problem.
  u_adj_incident = (omega**2) * (jnp.conj(params.G_sample) @ cotangent)

  rhs_adjoint = -(omega**2) * u_adj_incident * nu_vect.reshape((-1, 1))
  Sigma_adj = _solve_all_sources_adj(params, nu_vect, rhs_adjoint)

  u_adj = _apply_conj_green_all(params, Sigma_adj) + u_adj_incident

  return -jnp.sum(jnp.conj(u_total) * u_adj, axis=1) * params.cell_area


# ---------------------------------------------------------------------------
# The forward map.
# ---------------------------------------------------------------------------
@jit
def near_field_map(params: NearFieldParams, nu_vect: Array) -> Array:
  """Compute the near field on the observation circle.

  This is the plain forward map, with no custom differentiation rule attached.

  Args:
    params: Configuration for the near-field map.
    nu_vect: Contrast vector.

  Returns:
    The near field, of shape (n_theta, n_theta), indexed (source, receiver).
  """
  Sigma = _solve_all_sources(params, nu_vect, _incident_rhs(params, nu_vect))
  return _sample_near_field(params, Sigma)


@partial(custom_jvp, nondiff_argnums=(0,))
def near_field_map_forward_mode(
  params: NearFieldParams, nu_vect: Array
) -> Array:
  """The near-field map, differentiable in forward mode.

  Use with `jax.jvp` or `jax.linearize`; the tangent rule is the Born
  approximation.  For reverse mode use `near_field_map_reverse_mode` instead.

  Args:
    params: Configuration for the near-field map.
    nu_vect: Contrast vector.

  Returns:
    The near field, flattened to shape (n_theta * n_theta,).
  """
  Sigma = _solve_all_sources(params, nu_vect, _incident_rhs(params, nu_vect))
  return _sample_near_field(params, Sigma).reshape((-1,))


@near_field_map_forward_mode.defjvp
def near_field_map_forward_mode_jvp(
  params: NearFieldParams, primals: tuple[Array], tangents: tuple[Array]
) -> tuple[Array, Array]:
  """Born approximation: the forward-mode rule of the near-field map.

  Differentiating ``A(nu) sigma = f(nu)`` with ``f(nu) = -omega^2 nu u_i`` gives

    A delta_sigma = -omega^2 delta_nu * (G sigma + u_i),

  i.e. the tangent source is driven by the *total* field, unlike
  `jax_ls.ls_solver_field` where the right-hand side is held fixed.

  Args:
    params: Configuration for the near-field map.
    primals: Tuple of primal inputs (nu_vect,).
    tangents: Tuple of tangent inputs (delta_nu_vect,).

  Returns:
    A tuple of (primal output, tangent output), both flattened.
  """
  (nu_vect,) = primals
  (delta_nu_vect,) = tangents

  Sigma = _solve_all_sources(params, nu_vect, _incident_rhs(params, nu_vect))
  u_total = _total_field(params, Sigma)

  rhs_tangent = (
    -(params.ls_params.omega**2) * delta_nu_vect.reshape((-1, 1)) * u_total
  )
  delta_Sigma = _solve_all_sources(params, nu_vect, rhs_tangent)

  return (
    _sample_near_field(params, Sigma).reshape((-1,)),
    _sample_near_field(params, delta_Sigma).reshape((-1,)),
  )


@partial(custom_vjp, nondiff_argnums=(0,))
def near_field_map_reverse_mode(
  params: NearFieldParams, nu_vect: Array
) -> Array:
  """The near-field map, differentiable in reverse mode.

  Use with `jax.grad` or `jax.vjp`.  The rule implements the transpose ``J^T``,
  which is the convention JAX's reverse mode expects for complex outputs -- not
  the Hermitian adjoint ``J^H``.  For ``J^H`` use `born_adjoint`.

  Args:
    params: Configuration for the near-field map.
    nu_vect: Contrast vector.

  Returns:
    The near field, flattened to shape (n_theta * n_theta,).
  """
  Sigma = _solve_all_sources(params, nu_vect, _incident_rhs(params, nu_vect))
  return _sample_near_field(params, Sigma).reshape((-1,))


def near_field_map_reverse_mode_fwd(
  params: NearFieldParams, nu_vect: Array
) -> tuple[Array, tuple[Array, Array]]:
  """Forward pass of the reverse-mode rule."""
  Sigma = _solve_all_sources(params, nu_vect, _incident_rhs(params, nu_vect))
  return _sample_near_field(params, Sigma).reshape((-1,)), (Sigma, nu_vect)


def near_field_map_reverse_mode_bwd(
  params: NearFieldParams,
  residuals: tuple[Array, Array],
  cotangent: Array,
) -> tuple[Array]:
  """Backward pass of the reverse-mode rule.

  ``J^T w = conj(J^H conj(w))``, so the Hermitian adjoint is reused with the
  cotangent conjugated on the way in and the result conjugated on the way out.
  """
  Sigma, nu_vect = residuals
  n_theta = params.n_theta
  adjoint = _adjoint_apply(
    params, nu_vect, Sigma, jnp.conj(cotangent).reshape((n_theta, n_theta))
  )
  return (jnp.conj(adjoint),)


near_field_map_reverse_mode.defvjp(
  near_field_map_reverse_mode_fwd, near_field_map_reverse_mode_bwd
)


# ---------------------------------------------------------------------------
# Explicit linearisation and adjoint.
# ---------------------------------------------------------------------------
@jit
def born_linearization(
  params: NearFieldParams, nu_vect: Array, delta_nu_vect: Array
) -> Array:
  """Apply the Jacobian J of the near-field map (the Born approximation).

  Args:
    params: Configuration for the near-field map.
    nu_vect: Contrast vector the map is linearised about.
    delta_nu_vect: Perturbation of the contrast.

  Returns:
    J delta_nu, flattened to shape (n_theta * n_theta,).
  """
  _, tangent = near_field_map_forward_mode_jvp(
    params, (nu_vect,), (delta_nu_vect,)
  )
  return tangent


@jit
def born_adjoint(
  params: NearFieldParams, nu_vect: Array, cotangent: Array
) -> Array:
  """Apply the Hermitian adjoint J^H of the near-field map.

  Satisfies ``<w, J t> = <J^H w, t>`` for the Hermitian inner product
  ``<a, b> = sum(conj(a) * b)``.  This is the object a matrix-free least-squares
  solver needs; it is *not* what `jax.grad` expects (see
  `near_field_map_reverse_mode`).

  Args:
    params: Configuration for the near-field map.
    nu_vect: Contrast vector the map is linearised about.
    cotangent: Cotangent in data space, flat or of shape (n_theta, n_theta).

  Returns:
    J^H w, of shape (n_dof,).
  """
  Sigma = _solve_all_sources(params, nu_vect, _incident_rhs(params, nu_vect))
  n_theta = params.n_theta
  return _adjoint_apply(
    params, nu_vect, Sigma, cotangent.reshape((n_theta, n_theta))
  )


# ---------------------------------------------------------------------------
# Misfit.
# ---------------------------------------------------------------------------
@partial(custom_vjp, nondiff_argnums=(0, 1))
def near_field_l2_loss(
  params: NearFieldParams, u_data: Array, nu_vect: Array
) -> Array:
  """Compute the L2 misfit of the near-field map.

  Args:
    params: Configuration for the near-field map.
    u_data: Reference near-field data, of shape (n_theta, n_theta).
    nu_vect: Contrast vector.

  Returns:
    The scalar L2 loss 0.5 * ||F[nu] - u_data||^2.
  """
  Sigma = _solve_all_sources(params, nu_vect, _incident_rhs(params, nu_vect))
  residual = _sample_near_field(params, Sigma) - u_data
  return 0.5 * jnp.sum(jnp.abs(residual) ** 2)


def near_field_l2_loss_fwd(
  params: NearFieldParams, u_data: Array, nu_vect: Array
) -> tuple[Array, tuple[Array, Array, Array]]:
  """Forward pass of the L2 loss VJP."""
  Sigma = _solve_all_sources(params, nu_vect, _incident_rhs(params, nu_vect))
  residual = _sample_near_field(params, Sigma) - u_data
  loss = 0.5 * jnp.sum(jnp.abs(residual) ** 2)
  return loss, (Sigma, residual, nu_vect)


def near_field_l2_loss_bwd(
  params: NearFieldParams,
  u_data: Array,
  residuals: tuple[Array, Array, Array],
  cotangent: float,
) -> tuple[Array]:
  """Backward pass of the L2 loss VJP (the adjoint-state gradient).

  The gradient of a real-valued misfit is the real part of ``J^H`` applied to
  the residual, so the same adjoint solve serves both.
  """
  Sigma, residual, nu_vect = residuals
  return (
    cotangent * jnp.real(_adjoint_apply(params, nu_vect, Sigma, residual)),
  )


near_field_l2_loss.defvjp(near_field_l2_loss_fwd, near_field_l2_loss_bwd)


@jit
def near_field_l2_loss_and_grad(
  params: NearFieldParams, u_data: Array, nu_vect: Array
) -> tuple[Array, Array]:
  """Compute the L2 misfit and its gradient in a single pass.

  Equivalent to ``jax.value_and_grad(near_field_l2_loss, argnums=2)`` but
  written out directly, which avoids retracing the forward solve.

  Args:
    params: Configuration for the near-field map.
    u_data: Reference near-field data, of shape (n_theta, n_theta).
    nu_vect: Contrast vector.

  Returns:
    A tuple of (loss value, gradient vector).
  """
  loss, residuals = near_field_l2_loss_fwd(params, u_data, nu_vect)
  (gradient,) = near_field_l2_loss_bwd(params, u_data, residuals, 1.0)
  return loss, gradient


# ---------------------------------------------------------------------------
# Deprecated aliases.
#
# `near_field_map_vect_vjp` used to carry a `custom_vjp` implementing the
# Hermitian adjoint J^H, which made `jax.grad` through it silently wrong.  The
# alias now points at the reverse-mode map, which is correct under `jax.grad`;
# code that wanted J^H should call `born_adjoint`.
# ---------------------------------------------------------------------------
near_field_map_vect = near_field_map_forward_mode
near_field_map_vect_jax = near_field_map_reverse_mode
near_field_map_vect_vjp = near_field_map_reverse_mode
near_field_l2_loss_grad = near_field_l2_loss_and_grad
