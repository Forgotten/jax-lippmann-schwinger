"""Solving the inverse problem: the linearised subproblem, and the outer loop.

Three distinct iterations live in this package and they are easy to confuse,
because "GMRES" and "Gauss-Newton" sit at *different levels* and are not
alternatives to each other:

===========================  ==================================  ==============
Level                        Solves                              Default
===========================  ==================================  ==============
1. forward (`jax_ls.solvers`) ``A sigma = f``, square             GMRES
2. inverse, linear (here)    ``min ||J dnu - r||``, rectangular   GMRES on the
                                                                 normal equations
3. inverse, nonlinear (here) ``min loss(nu)``                     Gauss-Newton
===========================  ==================================  ==============

Level 2, and the trade-off between the options
----------------------------------------------
The Gauss-Newton step solves ``min ||J dnu - r||^2`` over the band-limited
subspace.  Three ways to do it are offered, and they are *not* equivalent:

* `NormalEquationsConfig` (default) forms ``J^H J dnu = J^H r`` implicitly and
  hands it to CG.  Conditioning ``kappa(J)^2``.
* `CglsConfig` works on ``J`` directly and converges according to ``kappa(J)``,
  which is strictly better on an ill-conditioned problem.

Note that *GMRES* on the normal equations is unavailable, for a structural
reason worth knowing: jax's gmres needs a transposable operator, and ours runs a
GMRES solve inside every application of ``J``.  CG is the correct method for an
Hermitian positive semi-definite operator anyway.  See `NormalEquationsConfig`.

The default is the normal equations because band-limiting (`jax_ls.filters`)
keeps ``kappa(J)`` small enough for the squaring to be affordable.  On a hard
problem, switch to CGLS.

Level 3
-------
`GaussNewton` converges in very few outer steps -- measured, one step from a
zero start cut the misfit tenfold -- and it searches only the band-limited
subspace, which is a twentyfold smaller space at ``omega = 8 pi``.  `QuasiNewton`
wraps ``jax.scipy.optimize.minimize`` and needs only the gradient, which makes it
the right tool for a *simultaneous* multi-frequency polish where there is no
single ``J`` to linearise, and the fallback when a Gauss-Newton line search
fails.
"""

from functools import partial
from typing import Callable, NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.optimize  # noqa: F401  (lazily loaded submodule)

from flax import struct

from .domain import Domain
from .filters import DEFAULT_BAND_FACTOR, band_limit
from .losses import FrequencyDataset, Weighting, make_wideband_loss
from .near_field import Linearization, linearize, near_field_l2_loss
from .sampling import PointSources, SamplingManifold
from .jax_ls import init_params

Array = jax.Array


# ---------------------------------------------------------------------------
# Level 2: the linearised subproblem.
# ---------------------------------------------------------------------------
@struct.dataclass
class CglsConfig:
  """Least squares on ``J`` directly; conditioning ``kappa(J)``.

  Attributes:
    n_iter: Maximum number of iterations.  CGLS terminates in at most as many
      steps as there are unknowns, so a larger budget is harmless -- see `rtol`.
    damp: Tikhonov weight added to the normal equations.
    rtol: Freeze the iteration once the normal-equation residual has dropped by
      this factor.  Not an optimisation but a *correctness* guard: past
      convergence ``gamma`` reaches round-off and the ratio ``gamma_new / gamma``
      that sets the search direction becomes meaningless, which makes plain CGLS
      diverge if it is run too long.  Measured on a 30x12 system, the error
      against a dense least-squares solve bottoms out at 3e-15 after 12 steps
      and reaches 3e+20 by 200 without this guard.
  """

  n_iter: int = struct.field(pytree_node=False, default=10)
  damp: float = struct.field(pytree_node=False, default=0.0)
  rtol: float = struct.field(pytree_node=False, default=1e-12)


@struct.dataclass
class NormalEquationsConfig:
  """Solve ``J^H J dnu = J^H r`` with a square-operator method.

  Conditioning is ``kappa(J)^2``; band-limiting is what makes that affordable.

  ``method="gmres"`` is **not available** for this operator, and the reason is
  structural rather than an omission.  ``jax.scipy.sparse.linalg.gmres`` routes
  through ``lax.custom_linear_solve``, which requires the operator to be
  transposable; ours applies ``J`` and ``J^H``, each of which runs a GMRES solve
  internally, and an inner ``custom_linear_solve`` cannot be transposed.  JAX
  raises ``NotImplementedError``.  ``cg`` has no such requirement and is in any
  case the right method here: ``J^H J`` is Hermitian positive semi-definite, so
  CG spans the same Krylov space as GMRES at lower cost per iteration.  GMRES
  remains the level-1 solver, where the operator is non-Hermitian and CG would
  not apply.

  Attributes:
    method: "cg".  See above for why "gmres" is rejected.
    n_iter: Maximum iterations.
    damp: Tikhonov weight added to the diagonal.
    tol: Relative tolerance.
  """

  method: str = struct.field(pytree_node=False, default="cg")
  n_iter: int = struct.field(pytree_node=False, default=20)
  damp: float = struct.field(pytree_node=False, default=0.0)
  tol: float = struct.field(pytree_node=False, default=1e-6)


def cgls(
  forward: Callable[[Array], Array],
  adjoint: Callable[[Array], Array],
  rhs: Array,
  x0: Array,
  config: CglsConfig = CglsConfig(),
) -> Array:
  """Matrix-free least squares: ``min ||A x - rhs||^2 + damp ||x||^2``.

  `forward` and `adjoint` must be a genuine adjoint pair with respect to the real
  inner product on the domain of `A`; that is what `band_limit` being an
  orthogonal projector guarantees when it is composed into them.

  Args:
    forward: Applies ``A``.
    adjoint: Applies ``A^*`` (real-valued, as the contrast is real).
    rhs: Right-hand side, in data space.
    x0: Initial guess, in model space.
    config: Iteration count and damping.

  Returns:
    The approximate least-squares solution.
  """
  x = x0
  residual = rhs - forward(x)
  gradient = adjoint(residual) - config.damp * x
  direction = gradient
  gamma = jnp.sum(jnp.square(gradient))
  # Convergence is judged against the initial gradient, so `rtol` means the same
  # thing whatever the scale of the problem.
  threshold = jnp.square(config.rtol) * jnp.maximum(
    gamma, jnp.finfo(gamma.dtype).tiny
  )

  def step(carry, _):
    x, residual, gradient, direction, gamma = carry

    # Once converged, freeze: every further update would be driven by round-off.
    active = gamma > threshold

    mapped = forward(direction)
    denominator = jnp.real(jnp.vdot(mapped, mapped)) + config.damp * jnp.sum(
      jnp.square(direction)
    )
    alpha = jnp.where(
      active & (denominator > 0),
      gamma / jnp.where(denominator > 0, denominator, 1.0),
      0.0,
    )

    x = x + alpha * direction
    residual = residual - alpha * mapped
    gradient = jnp.where(active, adjoint(residual) - config.damp * x, gradient)
    gamma_next = jnp.where(active, jnp.sum(jnp.square(gradient)), gamma)
    beta = jnp.where(
      active & (gamma > 0), gamma_next / jnp.where(gamma > 0, gamma, 1.0), 0.0
    )
    direction = jnp.where(active, gradient + beta * direction, direction)

    return (x, residual, gradient, direction, gamma_next), None

  (x, *_), _ = jax.lax.scan(
    step, (x, residual, gradient, direction, gamma), None, length=config.n_iter
  )
  return x


def _solve_normal_equations(
  forward: Callable[[Array], Array],
  adjoint: Callable[[Array], Array],
  rhs: Array,
  x0: Array,
  config: NormalEquationsConfig,
) -> Array:
  """Solve ``A^* A x = A^* rhs`` with GMRES or CG."""
  operator = lambda x: adjoint(forward(x)) + config.damp * x  # noqa: E731
  normal_rhs = adjoint(rhs)

  if config.method == "gmres":
    raise NotImplementedError(
      "GMRES cannot be used on the normal equations here: jax's gmres goes "
      "through lax.custom_linear_solve, which needs a transposable operator, "
      "and applying J or J^H runs an inner GMRES that cannot be transposed. "
      "Use method='cg' (J^H J is Hermitian positive semi-definite, so CG spans "
      "the same Krylov space more cheaply) or switch to CglsConfig."
    )
  if config.method != "cg":
    raise ValueError(f"method must be 'cg', got {config.method!r}")

  solution, _ = jax.scipy.sparse.linalg.cg(
    operator, normal_rhs, x0=x0, tol=config.tol, maxiter=config.n_iter
  )
  return solution


def gauss_newton_step(
  lin: Linearization,
  residual: Array,
  omega: float,
  inner: CglsConfig | NormalEquationsConfig = NormalEquationsConfig(),
  band_factor: float | None = DEFAULT_BAND_FACTOR,
) -> Array:
  """One Gauss-Newton update, optionally band-limited.

  Solves ``min ||J B dnu - residual||`` where ``B`` projects onto the modes the
  frequency can resolve.  ``B`` is an orthogonal projector, hence self-adjoint,
  so composing it into both operators keeps them an adjoint pair.

  Band-limiting is the only regularisation used by Borges, Gillman and Greengard
  (SIAM J. Imaging Sci. 10(2), 641-664, 2017; arXiv:1608.06871).  Passing
  ``band_factor=None`` disables it, which leaves ``J^H J`` singular on the modes
  the data cannot see -- supply ``damp > 0`` in that case.

  Args:
    lin: Linearisation of the near-field map about the current contrast; supplies
      ``J`` and ``J^H`` from a single cached forward solve.
    residual: ``data - F[nu]``, flattened.
    omega: Frequency, which sets the band limit.
    inner: Which linear least-squares solver to use.
    band_factor: Multiple of `omega` to retain, or None to keep every mode.

  Returns:
    The update ``dnu``, real and (unless disabled) band-limited.
  """
  domain = lin.params.ls_params.domain
  project = (
    (lambda v: v)
    if band_factor is None
    else partial(band_limit, domain, omega, factor=band_factor)
  )

  forward = lambda v: lin.jvp(project(v))  # noqa: E731
  # The contrast is real, so the adjoint of J restricted to real inputs is the
  # real part of J^H -- exactly the projection `near_field_l2_loss` performs.
  adjoint = lambda w: project(jnp.real(lin.vjp(w)))  # noqa: E731

  zero = jnp.zeros(domain.n_dof)
  if isinstance(inner, CglsConfig):
    update = cgls(forward, adjoint, residual, zero, inner)
  else:
    update = _solve_normal_equations(forward, adjoint, residual, zero, inner)

  return project(update)


# ---------------------------------------------------------------------------
# Level 3: the outer loop.
# ---------------------------------------------------------------------------
@struct.dataclass
class GaussNewton:
  """Band-limited Gauss-Newton.

  Attributes:
    n_steps: Outer iterations.
    inner: Linear least-squares solver for each step.
    band_factor: Multiple of `omega` retained in the search space, or None to
      disable band-limiting (see `gauss_newton_step`).
    backtrack: Whether to halve the step until the misfit decreases.
  """

  n_steps: int = struct.field(pytree_node=False, default=2)
  inner: CglsConfig | NormalEquationsConfig = struct.field(
    pytree_node=False, default=NormalEquationsConfig()
  )
  band_factor: float | None = struct.field(
    pytree_node=False, default=DEFAULT_BAND_FACTOR
  )
  backtrack: bool = struct.field(pytree_node=False, default=True)


@struct.dataclass
class QuasiNewton:
  """BFGS on the misfit, via `jax.scipy.optimize.minimize`.

  Needs only the gradient, so it works for the simultaneous multi-frequency loss
  where there is no single ``J``.

  Attributes:
    maxiter: Maximum iterations.
  """

  maxiter: int = struct.field(pytree_node=False, default=50)


Optimizer = GaussNewton | QuasiNewton


class InversionResult(NamedTuple):
  """Outcome of an inversion.

  Attributes:
    nu: The reconstructed contrast.
    loss: Final misfit.
    history: One record per outer step or frequency.
  """

  nu: Array
  loss: float
  history: list[dict]


def _backtracking(loss_fn, nu, update, loss_before, n_halvings=4):
  """Halve the step until the misfit decreases; return the accepted step."""
  step = 1.0
  for _ in range(n_halvings + 1):
    if float(loss_fn(nu + step * update)) < loss_before:
      return step
    step *= 0.5
  return 0.0


def minimize(
  params,
  data: Array,
  nu0: Array,
  optimizer: Optimizer = GaussNewton(),
) -> InversionResult:
  """Minimise the single-frequency misfit.

  Args:
    params: Near-field configuration.
    data: Measured near field, of shape `params.data_shape`.
    nu0: Initial contrast.
    optimizer: `GaussNewton` or `QuasiNewton`.

  Returns:
    An `InversionResult`.
  """
  loss_fn = jax.jit(partial(near_field_l2_loss, params, data))
  omega = float(params.ls_params.omega)

  if isinstance(optimizer, QuasiNewton):
    result = jax.scipy.optimize.minimize(
      loss_fn, x0=nu0, method="bfgs", options={"maxiter": optimizer.maxiter}
    )
    return InversionResult(
      nu=result.x,
      loss=float(loss_fn(result.x)),
      history=[
        {"iterations": int(result.nit), "loss": float(loss_fn(result.x))}
      ],
    )

  nu = nu0
  history = []
  for _ in range(optimizer.n_steps):
    lin = linearize(params, nu)
    loss_before = float(loss_fn(nu))
    residual = data.reshape((-1,)) - lin.data

    update = gauss_newton_step(
      lin, residual, omega, optimizer.inner, optimizer.band_factor
    )
    step = (
      _backtracking(loss_fn, nu, update, loss_before)
      if optimizer.backtrack
      else 1.0
    )
    nu = nu + step * update

    history.append(
      {"loss_before": loss_before, "loss": float(loss_fn(nu)), "step": step}
    )

  return InversionResult(nu=nu, loss=float(loss_fn(nu)), history=history)


# ---------------------------------------------------------------------------
# The frequency sweep.
# ---------------------------------------------------------------------------
@struct.dataclass
class SweepStage:
  """One frequency of a sweep, and how much work to spend on it.

  Attributes:
    omega: Angular frequency.
    max_updates: Maximum optimiser iterations at this frequency.  Continuation
      means each stage starts close to its own minimiser, so a small budget is
      usually enough; the last stage normally gets more.
    n_sources: Number of illuminations; defaults to the ``O(omega)`` rule.
    n_receivers: Number of receivers; defaults to the ``O(omega)`` rule.
  """

  omega: float = struct.field(pytree_node=False)
  max_updates: int = struct.field(pytree_node=False, default=20)
  n_sources: int | None = struct.field(pytree_node=False, default=None)
  n_receivers: int | None = struct.field(pytree_node=False, default=None)


def as_stages(spec) -> list[SweepStage]:
  """Coerce a sweep specification into a list of `SweepStage`, ascending.

  Accepts whatever is most natural at the call site:

  * ``[SweepStage(...), ...]``
  * ``[{"omega": 6.0, "max_updates": 20}, ...]``
  * ``[(6.0, 20), (12.0, 20), ...]``
  * ``{6.0: 20, 12.0: 30}`` -- frequency to update budget
  * ``[6.0, 12.0, 24.0]`` -- bare frequencies, default budget

  Args:
    spec: One of the forms above.

  Returns:
    The stages, sorted by increasing frequency, because a sweep must start low.
  """
  if isinstance(spec, dict):
    items = [
      SweepStage(omega=float(w), max_updates=int(n)) for w, n in spec.items()
    ]
  else:
    items = []
    for entry in spec:
      if isinstance(entry, SweepStage):
        items.append(entry)
      elif isinstance(entry, dict):
        items.append(SweepStage(**entry))
      elif isinstance(entry, (tuple, list)):
        omega, updates = entry
        items.append(SweepStage(omega=float(omega), max_updates=int(updates)))
      else:
        items.append(SweepStage(omega=float(entry)))
  return sorted(items, key=lambda stage: stage.omega)


@struct.dataclass
class FrequencySchedule:
  """Builders for the frequencies of a sweep, and the sampling at each.

  The sampling rules follow Borges, Gillman and Greengard (SIAM J. Imaging Sci.
  10(2), 641-664, 2017; arXiv:1608.06871), rescaled from their side-``pi``
  domain: ``N_src ~ 2 omega a / pi`` illuminations and ``N_recv ~ 4 omega a / pi``
  receivers, both ``O(omega)``.  Since the cost of a solve is *linear* in the
  number of sources, honouring this makes the low-frequency end nearly free.

  Attributes:
    omegas: The frequencies, ascending.
  """

  omegas: Array

  @classmethod
  def geometric(
    cls, omega_min: float, omega_max: float, ratio: float = 1.3
  ) -> "FrequencySchedule":
    """Geometrically spaced frequencies.

    Sirgue and Pratt (2004): with a range of source-receiver offsets the
    wavenumber coverage of neighbouring frequencies overlaps redundantly, so a
    sparse set suffices.
    """
    count = max(
      2, int(np.ceil(np.log(omega_max / omega_min) / np.log(ratio))) + 1
    )
    return cls(omegas=jnp.asarray(np.geomspace(omega_min, omega_max, count)))

  @classmethod
  def uniform_band(
    cls,
    domain: Domain,
    omega_min: float,
    omega_max: float,
    band_step: float = 0.5,
  ) -> "FrequencySchedule":
    """Frequencies that open the band by `band_step` modes at a time.

    The conservative schedule of Borges et al., who advance the band index by
    about half a mode per step.
    """
    d_omega = float(jnp.pi * band_step / domain.ax)
    count = max(2, int(np.ceil((omega_max - omega_min) / d_omega)) + 1)
    return cls(omegas=jnp.linspace(omega_min, omega_max, count))

  def __len__(self) -> int:
    return self.omegas.shape[0]

  def n_sources_for(self, domain: Domain, omega: float) -> int:
    """``ceil(2 omega ax / pi)``, at least 4."""
    return int(max(4, np.ceil(2 * float(omega) * domain.ax / np.pi)))

  def n_receivers_for(self, domain: Domain, omega: float) -> int:
    """``ceil(4 omega ax / pi)``, at least 8."""
    return int(max(8, np.ceil(4 * float(omega) * domain.ax / np.pi)))

  def stages(self, max_updates: int | list[int] = 20) -> list[SweepStage]:
    """Turn the frequencies into sweep stages.

    Args:
      max_updates: A single budget for every stage, or one per stage.

    Returns:
      The stages, ascending.
    """
    budgets = (
      [int(max_updates)] * len(self)
      if isinstance(max_updates, int)
      else [int(n) for n in max_updates]
    )
    if len(budgets) != len(self):
      raise ValueError(
        f"got {len(budgets)} update budgets for {len(self)} frequencies"
      )
    return [
      SweepStage(omega=float(omega), max_updates=budget)
      for omega, budget in zip(self.omegas, budgets)
    ]


def make_sweep_params(
  domain: Domain,
  omega: float,
  radius: float,
  n_sources: int | None = None,
  n_receivers: int | None = None,
  **kwargs,
):
  """Near-field parameters for one frequency of a sweep.

  Sources and receivers are separate manifolds on the same circle, sized by the
  ``O(omega)`` rule unless given explicitly.

  Args:
    domain: The shared grid.
    omega: Angular frequency.
    radius: Radius of the observation circle.
    n_sources: Override for the number of illuminations.
    n_receivers: Override for the number of receivers.
    **kwargs: Forwarded to `init_params`.

  Returns:
    A `NearFieldParams` for this frequency.
  """
  from .near_field import make_near_field_params

  rule = FrequencySchedule(omegas=jnp.asarray([omega]))
  n_src = n_sources or rule.n_sources_for(domain, omega)
  n_rec = n_receivers or rule.n_receivers_for(domain, omega)

  ls_params = init_params(
    domain.ax, domain.ay, domain.n, domain.m, float(omega), **kwargs
  )
  return make_near_field_params(
    ls_params,
    PointSources(SamplingManifold.circle(radius, n_src)),
    SamplingManifold.circle(radius, n_rec),
  )


def frequency_sweep(
  domain: Domain,
  stages,
  data_for: Callable,
  radius: float,
  nu0: Array | None = None,
  optimizer: Optimizer | None = None,
  band_factor: float | None = None,
  verbose: bool = True,
) -> InversionResult:
  """Invert by sweeping frequency from low to high.

  Each stage is solved starting from the previous stage's reconstruction.  That
  is the whole idea: the low-frequency misfit is smooth and nearly convex but
  poorly resolved, the high-frequency one is sharply resolved but has a basin of
  attraction only about half a wavelength wide, and continuation walks the
  iterate from one to the other.  This is *recursive linearization* (Chen;
  Bao & Li; Borges, Gillman & Greengard 2017) and the multiscale argument of
  Bunks et al. (1995).

  The outer optimiser defaults to BFGS, which needs only the gradient the package
  already provides correctly.  Pass a `GaussNewton` to use the band-limited
  Gauss-Newton step instead.

  Args:
    domain: The shared grid, sized for the *highest* frequency.  One fixed grid
      is used throughout, so the contrast never needs interpolating between
      stages.
    stages: The sweep, in any form `as_stages` accepts -- most simply a list of
      ``(omega, max_updates)`` pairs or a ``{omega: max_updates}`` dict.
    data_for: Called as ``data_for(params)``, returning the measured near field
      at that frequency.  Each frequency has its own kernel, incident field and
      sampling geometry, so data is per-frequency and its shape changes with the
      sampling rule.
    radius: Radius of the observation circle.
    nu0: Initial contrast; zero if omitted.
    optimizer: Outer optimiser.  Defaults to ``QuasiNewton(max_updates)`` per
      stage, i.e. BFGS with that stage's budget.
    band_factor: If given, project the estimate onto ``|xi| <= band_factor *
      omega`` when entering each stage, discarding what the previous frequency
      could not have known.  ``None`` (the default) leaves the estimate alone;
      see `jax_ls.filters`.
    verbose: Whether to print progress.

  Returns:
    An `InversionResult` whose history has one record per frequency.
  """
  stage_list = as_stages(stages)
  nu = jnp.zeros(domain.n_dof) if nu0 is None else nu0
  history = []
  loss = float("nan")

  for stage in stage_list:
    params = make_sweep_params(
      domain, stage.omega, radius, stage.n_sources, stage.n_receivers
    )
    data = data_for(params)

    if band_factor is not None:
      nu = band_limit(domain, stage.omega, nu, factor=band_factor)

    step_optimizer = optimizer or QuasiNewton(maxiter=stage.max_updates)
    result = minimize(params, data, nu, step_optimizer)
    nu, loss = result.nu, result.loss

    history.append(
      {
        "omega": stage.omega,
        "n_sources": params.n_sources,
        "n_receivers": params.n_receivers,
        "max_updates": stage.max_updates,
        "loss": loss,
        "steps": result.history,
      }
    )
    if verbose:
      print(
        f"  omega {stage.omega:7.2f}  sources {params.n_sources:3d}  "
        f"receivers {params.n_receivers:3d}  misfit {loss:.4e}"
      )

  return InversionResult(nu=nu, loss=loss, history=history)


# The name used in the inverse-scattering literature.
recursive_linearization = frequency_sweep


def wideband_minimize(
  datasets: list[FrequencyDataset],
  nu0: Array,
  optimizer: QuasiNewton = QuasiNewton(),
  policy: Weighting = Weighting.DATA_NORM,
) -> InversionResult:
  """Minimise the *simultaneous* multi-frequency misfit.

  All frequencies are active at once, so the weighting of `policy` matters here
  -- unlike in `frequency_sweep`, where scaling a single-frequency objective
  changes neither its minimiser nor the step.  Best used as a final polish over
  the top few frequencies.

  Args:
    datasets: One `FrequencyDataset` per frequency.
    nu0: Initial contrast.
    optimizer: BFGS settings; the loss has no single ``J`` to linearise.
    policy: How to weight the frequencies.

  Returns:
    An `InversionResult`.
  """
  loss_fn = jax.jit(make_wideband_loss(datasets, policy))
  result = jax.scipy.optimize.minimize(
    loss_fn, x0=nu0, method="bfgs", options={"maxiter": optimizer.maxiter}
  )
  return InversionResult(
    nu=result.x,
    loss=float(loss_fn(result.x)),
    history=[{"iterations": int(result.nit), "loss": float(loss_fn(result.x))}],
  )
