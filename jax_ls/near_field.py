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

import jax
import jax.numpy as jnp
from jax import custom_jvp, custom_vjp, jit, vmap

from flax import struct

from .sampling import IncidentField, PointSources, SamplingManifold
from .jax_ls import LippSchwinParams, init_params
from .jax_ls import apply_conj_green_function, apply_green_function
from .jax_ls import apply_lipp_schwin, apply_lipp_schwin_adj
from .jax_ls import ls_solver, ls_solver_adj
from .solvers import BlockGmresConfig, block_solve

Array = jax.Array


@struct.dataclass
class NearFieldParams:
  """Configuration for the near-field map and its derivatives.

  The incident field and the sampling kernel are separate: sources and
  receivers may differ in number, position and kind.  When they coincide the
  two matrices are equal and `G_sample` returns either.

  Attributes:
    ls_params: Lippmann-Schwinger configuration.
    incident_field: The incident field of every source, of shape
      ``(n_dof, n_sources)``.
    sampling_kernel: The kernel sampling the scattered field at the receivers,
      of shape ``(n_dof, n_receivers)``.
    block: Optional block-GMRES settings; `None` solves each source separately.
  """

  ls_params: LippSchwinParams
  incident_field: Array
  sampling_kernel: Array
  # When set, the per-source solves are replaced by one block GMRES solve over
  # all right-hand sides at once.  Static, so switching it is a recompile rather
  # than a traced branch.  See `jax_ls.solvers.BlockGmresConfig` for when that
  # is a good idea -- on CPU it usually is not.
  block: BlockGmresConfig | None = struct.field(pytree_node=False, default=None)

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
  def n_sources(self) -> int:
    """Number of illuminations."""
    return self.incident_field.shape[1]

  @property
  def n_receivers(self) -> int:
    """Number of measurement points."""
    return self.sampling_kernel.shape[1]

  @property
  def data_shape(self) -> tuple[int, int]:
    """Shape of the near field, indexed (source, receiver)."""
    return (self.n_sources, self.n_receivers)

  @property
  def is_reciprocal(self) -> bool:
    """Whether sources and receivers coincide.

    Only then is the near-field matrix symmetric, and only then is `G_sample`
    meaningful.
    """
    return self.incident_field.shape == self.sampling_kernel.shape and bool(
      jnp.array_equal(self.incident_field, self.sampling_kernel)
    )

  @property
  def n_theta(self) -> int:
    """Deprecated: the common number of sources and receivers."""
    if self.n_sources != self.n_receivers:
      raise ValueError(
        f"`n_theta` is ambiguous with {self.n_sources} sources and "
        f"{self.n_receivers} receivers; use `n_sources` or `n_receivers`."
      )
    return self.n_receivers

  @property
  def G_sample(self) -> Array:
    """Deprecated: the sampling kernel, which used to serve both roles."""
    return self.sampling_kernel


def make_near_field_params(
  ls_params: LippSchwinParams,
  incident: IncidentField,
  receivers: SamplingManifold,
  block: BlockGmresConfig | None = None,
) -> NearFieldParams:
  """Assemble near-field parameters from an illumination and a receiver set.

  This is the general constructor; `init_params_near_field` is the convenience
  wrapper that puts point sources and receivers on the same circle.

  Args:
    ls_params: Lippmann-Schwinger configuration.
    incident: What illuminates the medium.
    receivers: Where the scattered field is measured.
    block: Optional block-GMRES settings for the per-source solves.

  Returns:
    An initialized NearFieldParams object.
  """
  domain = ls_params.domain
  omega = ls_params.omega
  return NearFieldParams(
    ls_params=ls_params,
    incident_field=incident.evaluate(domain, omega),
    sampling_kernel=receivers.sampling_kernel(domain, omega),
    block=block,
  )


def init_params_near_field(
  ax: float,
  ay: float,
  n: int,
  m: int,
  r: float,
  n_theta: int,
  omega: float,
  n_receivers: int | None = None,
  block: BlockGmresConfig | None = None,
  **kwargs,
) -> NearFieldParams:
  """Initialize parameters for the near-field measurement system.

  Point sources and receivers on a circle of radius `r`.  By default they
  coincide, which makes the near-field matrix symmetric by reciprocity; pass
  `n_receivers` to separate them.

  Args:
    ax: Length of the domain in the x direction.
    ay: Length of the domain in the y direction.
    n: Number of discretization points in the x direction.
    m: Number of discretization points in the y direction.
    r: Radius of the observation manifold.
    n_theta: Number of sources on the observation manifold.
    omega: Angular frequency.
    n_receivers: Number of receivers; defaults to `n_theta`.
    block: Optional block-GMRES settings for the per-source solves.
    **kwargs: Forwarded to `init_params` (e.g. `pad`, `truncation_radius`).

  Returns:
    An initialized NearFieldParams object.
  """
  ls_params = init_params(ax, ay, n, m, omega, **kwargs)

  sources = SamplingManifold.circle(r, n_theta)
  receivers = (
    sources
    if n_receivers is None or n_receivers == n_theta
    else SamplingManifold.circle(r, n_receivers)
  )

  return make_near_field_params(
    ls_params, PointSources(sources), receivers, block=block
  )


# ---------------------------------------------------------------------------
# Building blocks.  Every public function below is assembled from these, so the
# forward map, its linearisation and its adjoint cannot drift apart.
# ---------------------------------------------------------------------------
def _incident_rhs(params: NearFieldParams, nu_vect: Array) -> Array:
  """Right-hand side of the integral equation, one column per source."""
  return (
    -(params.ls_params.omega**2)
    * nu_vect.reshape((-1, 1))
    * params.incident_field
  )


def _solve_all_sources(
  params: NearFieldParams, nu_vect: Array, rhs: Array
) -> Array:
  """Solve the integral equation for every column of `rhs`."""
  if params.block is not None:
    operator = vmap(
      partial(apply_lipp_schwin, params.ls_params, nu_vect),
      in_axes=1,
      out_axes=1,
    )
    return block_solve(operator, rhs, params.block)

  solve = vmap(
    partial(ls_solver, params.ls_params, nu_vect), in_axes=1, out_axes=1
  )
  return solve(rhs)


def _solve_all_sources_adj(
  params: NearFieldParams, nu_vect: Array, rhs: Array
) -> Array:
  """Solve the adjoint integral equation for every column of `rhs`."""
  if params.block is not None:
    operator = vmap(
      partial(apply_lipp_schwin_adj, params.ls_params, nu_vect),
      in_axes=1,
      out_axes=1,
    )
    return block_solve(operator, rhs, params.block)

  solve = vmap(
    partial(ls_solver_adj, params.ls_params, nu_vect), in_axes=1, out_axes=1
  )
  return solve(rhs)


def _apply_green_all(params: NearFieldParams, sigma: Array) -> Array:
  """Apply the Green's function to every column of `sigma`."""
  green = vmap(
    partial(apply_green_function, params.ls_params), in_axes=1, out_axes=1
  )
  return green(sigma)


def _apply_conj_green_all(params: NearFieldParams, sigma: Array) -> Array:
  """Apply the conjugate Green's function to every column of `sigma`."""
  green = vmap(
    partial(apply_conj_green_function, params.ls_params), in_axes=1, out_axes=1
  )
  return green(sigma)


def _total_field(params: NearFieldParams, sigma: Array) -> Array:
  """Total field (incident plus scattered) for every source."""
  return _apply_green_all(params, sigma) + params.incident_field


def _sample_near_field(params: NearFieldParams, sigma: Array) -> Array:
  """Sample the scattered field on the observation manifold.

  The result is indexed (source, receiver).  The minus sign is the ``-G``
  convention described in the module docstring.
  """
  return -(sigma.T @ params.sampling_kernel) * params.cell_area


def _adjoint_apply(
  params: NearFieldParams, nu_vect: Array, u_total: Array, cotangent: Array
) -> Array:
  """Apply the Hermitian adjoint J^H to a cotangent matrix.

  The adjoint problem is assembled in exactly the same "incident plus scattered"
  form as the forward problem, with the conjugate kernel:

    u_i^H  = omega^2 conj(S)^T-contracted w   (field radiated back by the residual)
    rhs^H  = -omega^2 nu * u_i^H
    sigma^H = A_adj^{-1} rhs^H
    u^H    = G^H sigma^H + u_i^H
    J^H w  = -cell_area * sum_sources conj(u_total) * u^H

  The cotangent is contracted along its *receiver* index against the sampling
  kernel, which leaves the source index to pair with the total field.  When
  sources and receivers coincide the opposite pairing gives the same answer,
  because the range of ``J`` consists of symmetric matrices -- the Born
  derivative is ``-omega^2 int u^s dnu u^r``, symmetric in (s, r) -- so ``J``
  annihilates the antisymmetric part of any cotangent.  That shortcut is not
  available once the two manifolds differ, which is why the transpose below is
  explicit.

  Args:
    params: Configuration for the near-field map.
    nu_vect: Contrast vector.
    u_total: Total field of every source, of shape (n_dof, n_sources).
    cotangent: Cotangent, of shape (n_sources, n_receivers).

  Returns:
    The cotangent with respect to `nu_vect`, of shape (n_dof,).
  """
  omega = params.ls_params.omega

  # Incident field of the adjoint problem.
  # `cotangent` is indexed (source, receiver) and `sampling_kernel` is
  # (n_dof, n_receivers), so the transpose is what contracts the receiver
  # index and leaves the source index -- giving (n_dof, n_sources), which is
  # what the total field is indexed by.  When sources and receivers coincide
  # the transpose is invisible, because the range of J consists of symmetric
  # matrices; it stops being invisible the moment the two differ.
  u_adj_incident = (omega**2) * (jnp.conj(params.sampling_kernel) @ cotangent.T)

  rhs_adjoint = -(omega**2) * u_adj_incident * nu_vect.reshape((-1, 1))
  sigma_adj = _solve_all_sources_adj(params, nu_vect, rhs_adjoint)

  u_adj = _apply_conj_green_all(params, sigma_adj) + u_adj_incident

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
    The near field, of shape (n_sources, n_receivers).
  """
  sigma = _solve_all_sources(params, nu_vect, _incident_rhs(params, nu_vect))
  return _sample_near_field(params, sigma)


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
    The near field, flattened to shape (n_sources * n_receivers,).
  """
  sigma = _solve_all_sources(params, nu_vect, _incident_rhs(params, nu_vect))
  return _sample_near_field(params, sigma).reshape((-1,))


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

  sigma = _solve_all_sources(params, nu_vect, _incident_rhs(params, nu_vect))
  u_total = _total_field(params, sigma)

  rhs_tangent = (
    -(params.ls_params.omega**2) * delta_nu_vect.reshape((-1, 1)) * u_total
  )
  delta_sigma = _solve_all_sources(params, nu_vect, rhs_tangent)

  return (
    _sample_near_field(params, sigma).reshape((-1,)),
    _sample_near_field(params, delta_sigma).reshape((-1,)),
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
    The near field, flattened to shape (n_sources * n_receivers,).
  """
  sigma = _solve_all_sources(params, nu_vect, _incident_rhs(params, nu_vect))
  return _sample_near_field(params, sigma).reshape((-1,))


def near_field_map_reverse_mode_fwd(
  params: NearFieldParams, nu_vect: Array
) -> tuple[Array, tuple[Array, Array]]:
  """Forward pass of the reverse-mode rule."""
  sigma = _solve_all_sources(params, nu_vect, _incident_rhs(params, nu_vect))
  return (
    _sample_near_field(params, sigma).reshape((-1,)),
    (_total_field(params, sigma), nu_vect),
  )


def near_field_map_reverse_mode_bwd(
  params: NearFieldParams,
  residuals: tuple[Array, Array],
  cotangent: Array,
) -> tuple[Array]:
  """Backward pass of the reverse-mode rule.

  ``J^T w = conj(J^H conj(w))``, so the Hermitian adjoint is reused with the
  cotangent conjugated on the way in and the result conjugated on the way out.
  """
  u_total, nu_vect = residuals
  adjoint = _adjoint_apply(
    params, nu_vect, u_total, jnp.conj(cotangent).reshape(params.data_shape)
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
    J delta_nu, flattened to shape (n_sources * n_receivers,).
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
    cotangent: Cotangent in data space, flat or of shape
      (n_sources, n_receivers).

  Returns:
    J^H w, of shape (n_dof,).
  """
  sigma = _solve_all_sources(params, nu_vect, _incident_rhs(params, nu_vect))
  return _adjoint_apply(
    params,
    nu_vect,
    _total_field(params, sigma),
    cotangent.reshape(params.data_shape),
  )


# ---------------------------------------------------------------------------
# Cached linearisation.
# ---------------------------------------------------------------------------
@struct.dataclass
class Linearization:
  """``J`` and ``J^H`` about a fixed contrast, sharing one forward solve.

  `born_linearization` and `born_adjoint` each re-solve the forward problem
  internally, so a Gauss-Newton inner loop that applies them ``2 * n_iter``
  times pays for that solve every single time even though the linearisation
  point never moves.  Measured, the repeat is about half the cost of each
  application.  Building a `Linearization` once and calling `jvp` / `vjp` off it
  does the forward solve exactly once.

  Attributes:
    params: Configuration for the near-field map.
    nu_vect: The contrast the map is linearised about.
    sigma: Forward densities, one column per source.
    u_total: Total field of every source, of shape (n_dof, n_sources).
    data: The forward map at `nu_vect`, flattened -- free, since the forward
      solve had to happen anyway, and it is what the residual is measured from.
  """

  params: NearFieldParams
  nu_vect: Array
  sigma: Array
  u_total: Array
  data: Array

  def jvp(self, delta_nu_vect: Array) -> Array:
    """Apply ``J``: the Born approximation about `nu_vect`.

    Costs one batched solve, against two for `born_linearization`.

    Args:
      delta_nu_vect: Perturbation of the contrast.

    Returns:
      ``J delta_nu``, flattened.
    """
    rhs = (
      -(self.params.ls_params.omega**2)
      * delta_nu_vect.reshape((-1, 1))
      * self.u_total
    )
    delta_sigma = _solve_all_sources(self.params, self.nu_vect, rhs)
    return _sample_near_field(self.params, delta_sigma).reshape((-1,))

  def vjp(self, cotangent: Array) -> Array:
    """Apply the Hermitian adjoint ``J^H``.

    Costs one batched adjoint solve, against a forward *and* an adjoint solve
    for `born_adjoint`.  Deliberately a method rather than an autodiff rule:
    ``J^H`` is not what `jax.grad` expects, and calling it a `vjp` here is safe
    only because nothing differentiates through this object.

    Args:
      cotangent: Cotangent in data space, flat or of shape
        (n_sources, n_receivers).

    Returns:
      ``J^H w``, of shape (n_dof,).
    """
    return _adjoint_apply(
      self.params,
      self.nu_vect,
      self.u_total,
      cotangent.reshape(self.params.data_shape),
    )


@jit
def linearize(params: NearFieldParams, nu_vect: Array) -> Linearization:
  """Linearise the near-field map about a contrast.

  Does the forward solve once; `Linearization.jvp` and `Linearization.vjp` then
  reuse it.

  Args:
    params: Configuration for the near-field map.
    nu_vect: Contrast vector to linearise about.

  Returns:
    A `Linearization` about `nu_vect`.
  """
  sigma = _solve_all_sources(params, nu_vect, _incident_rhs(params, nu_vect))
  return Linearization(
    params=params,
    nu_vect=nu_vect,
    sigma=sigma,
    u_total=_total_field(params, sigma),
    data=_sample_near_field(params, sigma).reshape((-1,)),
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
    u_data: Reference near-field data, of shape (n_sources, n_receivers).
    nu_vect: Contrast vector.

  Returns:
    The scalar L2 loss 0.5 * ||F[nu] - u_data||^2.
  """
  sigma = _solve_all_sources(params, nu_vect, _incident_rhs(params, nu_vect))
  residual = _sample_near_field(params, sigma) - u_data
  return 0.5 * jnp.sum(jnp.abs(residual) ** 2)


def near_field_l2_loss_fwd(
  params: NearFieldParams, u_data: Array, nu_vect: Array
) -> tuple[Array, tuple[Array, Array, Array]]:
  """Forward pass of the L2 loss VJP."""
  sigma = _solve_all_sources(params, nu_vect, _incident_rhs(params, nu_vect))
  residual = _sample_near_field(params, sigma) - u_data
  loss = 0.5 * jnp.sum(jnp.abs(residual) ** 2)
  return loss, (_total_field(params, sigma), residual, nu_vect)


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
  u_total, residual, nu_vect = residuals
  return (
    cotangent * jnp.real(_adjoint_apply(params, nu_vect, u_total, residual)),
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
    u_data: Reference near-field data, of shape (n_sources, n_receivers).
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
