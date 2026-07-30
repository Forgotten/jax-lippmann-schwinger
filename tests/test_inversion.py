"""Unit tests for the inverse solvers and the frequency sweep.

Deliberately not covered here, because it already is elsewhere: the band-limiting
projector (`test_filters.py`), the gradient of the misfit (`test_derivatives.py`)
and the cached linearisation (`test_derivatives.py::TestLinearization`).
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import jax_ls
from jax_ls.domain import Domain
from jax_ls.inversion import (
  CglsConfig,
  NormalEquationsConfig,
  SweepStage,
  as_stages,
  cgls,
)

jax.config.update("jax_enable_x64", True)


class TestCgls:
  """`cgls` against a dense reference."""

  def test_matches_a_dense_least_squares_solve(self):
    """On a small overdetermined system, CGLS must reach `lstsq`."""
    key = jax.random.PRNGKey(0)
    key_a, key_b = jax.random.split(key)
    matrix = jax.random.normal(key_a, (30, 12))
    rhs = jax.random.normal(key_b, (30,))

    forward = lambda x: matrix @ x  # noqa: E731
    adjoint = lambda y: matrix.T @ y  # noqa: E731

    got = cgls(forward, adjoint, rhs, jnp.zeros(12), CglsConfig(n_iter=60))
    want = jnp.linalg.lstsq(matrix, rhs)[0]

    err = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
    assert err < 1e-8, f"CGLS vs lstsq: {err:.2e}"

  def test_does_not_diverge_when_run_past_convergence(self):
    """Iterating well past convergence must not destroy the answer.

    CGLS terminates in at most as many steps as there are unknowns.  Beyond
    that, ``gamma`` is round-off and the ratio setting the search direction is
    meaningless: unguarded, this system reaches a relative error of 3e+20 by 200
    iterations.  A caller who picks a generous `n_iter` must not be punished.
    """
    key_a, key_b = jax.random.split(jax.random.PRNGKey(0))
    matrix = jax.random.normal(key_a, (30, 12))
    rhs = jax.random.normal(key_b, (30,))
    forward = lambda x: matrix @ x  # noqa: E731
    adjoint = lambda y: matrix.T @ y  # noqa: E731
    want = jnp.linalg.lstsq(matrix, rhs)[0]

    for n_iter in (12, 60, 400):
      got = cgls(
        forward, adjoint, rhs, jnp.zeros(12), CglsConfig(n_iter=n_iter)
      )
      err = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
      assert err < 1e-10, f"n_iter={n_iter} diverged: {err:.2e}"

  def test_damping_shrinks_the_solution(self):
    """Tikhonov damping must bias towards zero."""
    key_a, key_b = jax.random.split(jax.random.PRNGKey(1))
    matrix = jax.random.normal(key_a, (20, 8))
    rhs = jax.random.normal(key_b, (20,))
    forward = lambda x: matrix @ x  # noqa: E731
    adjoint = lambda y: matrix.T @ y  # noqa: E731

    plain = cgls(forward, adjoint, rhs, jnp.zeros(8), CglsConfig(n_iter=40))
    damped = cgls(
      forward, adjoint, rhs, jnp.zeros(8), CglsConfig(n_iter=40, damp=10.0)
    )

    assert float(jnp.linalg.norm(damped)) < float(jnp.linalg.norm(plain))


class TestStages:
  """The sweep specification, in the forms a caller might reach for."""

  @pytest.mark.parametrize(
    "spec",
    [
      {20.0: 5, 10.0: 3},
      [(20.0, 5), (10.0, 3)],
      [{"omega": 20.0, "max_updates": 5}, {"omega": 10.0, "max_updates": 3}],
      [
        SweepStage(omega=20.0, max_updates=5),
        SweepStage(omega=10.0, max_updates=3),
      ],
    ],
  )
  def test_all_forms_agree_and_sort_ascending(self, spec):
    """A sweep must start at the lowest frequency however it was specified."""
    stages = as_stages(spec)
    assert [(s.omega, s.max_updates) for s in stages] == [(10.0, 3), (20.0, 5)]

  def test_bare_frequencies_get_a_default_budget(self):
    stages = as_stages([8.0, 4.0])
    assert [s.omega for s in stages] == [4.0, 8.0]
    assert all(s.max_updates > 0 for s in stages)


class TestFrequencySchedule:
  def test_sampling_rules_are_linear_in_frequency(self):
    """N_src ~ 2 omega a / pi and N_recv ~ 4 omega a / pi."""
    domain = Domain(1.0, 1.0, 64, 64)
    schedule = jax_ls.FrequencySchedule(omegas=jnp.asarray([10.0, 20.0, 40.0]))

    sources = [schedule.n_sources_for(domain, w) for w in (10.0, 20.0, 40.0)]
    receivers = [
      schedule.n_receivers_for(domain, w) for w in (10.0, 20.0, 40.0)
    ]

    assert sources == [int(np.ceil(2 * w / np.pi)) for w in (10.0, 20.0, 40.0)]
    assert receivers == [
      int(np.ceil(4 * w / np.pi)) for w in (10.0, 20.0, 40.0)
    ]
    # receivers are the more densely sampled of the two
    assert all(r > s for r, s in zip(receivers, sources))

  def test_geometric_spans_the_requested_range(self):
    schedule = jax_ls.FrequencySchedule.geometric(4.0, 40.0, ratio=1.5)
    assert float(schedule.omegas[0]) == pytest.approx(4.0)
    assert float(schedule.omegas[-1]) == pytest.approx(40.0)
    assert bool(jnp.all(jnp.diff(schedule.omegas) > 0))

  def test_stages_rejects_a_mismatched_budget_list(self):
    schedule = jax_ls.FrequencySchedule.geometric(4.0, 40.0, ratio=1.5)
    with pytest.raises(ValueError, match="update budgets"):
      schedule.stages([10, 20])


class TestNormalEquations:
  def test_gmres_is_refused_with_an_explanation(self):
    """GMRES cannot be used here, and the error must say why.

    jax's gmres needs a transposable operator; applying J runs a GMRES solve
    internally, which cannot be transposed.  Silently falling back would hide a
    real constraint.
    """
    params = jax_ls.init_params_near_field(1.0, 1.0, 24, 24, 1.5, 8, 10.0)
    contrast = params.ls_params.domain.sample(
      lambda X, Y: 0.3 * jnp.exp(-100 * (jnp.square(X) + jnp.square(Y)))
    )
    lin = jax_ls.linearize(params, contrast)

    with pytest.raises(NotImplementedError, match="transposable"):
      jax_ls.gauss_newton_step(
        lin,
        jnp.zeros(params.n_sources * params.n_receivers, dtype=complex),
        10.0,
        NormalEquationsConfig(method="gmres"),
      )


class TestOuterLoops:
  """Both optimisers must reduce the misfit on the same problem."""

  @pytest.fixture(scope="class")
  def problem(self):
    params = jax_ls.init_params_near_field(1.0, 1.0, 32, 32, 1.5, 10, 12.0)
    truth = params.ls_params.domain.sample(
      lambda X, Y: (
        0.8 * jnp.exp(-200 * (jnp.square(X - 0.05) + jnp.square(Y + 0.05)))
      )
    )
    return params, truth, jax_ls.near_field_map(params, truth)

  def test_gauss_newton_reduces_the_misfit(self, problem):
    params, _, data = problem
    start = jnp.zeros(params.ls_params.n_dof)
    before = float(jax_ls.near_field_l2_loss(params, data, start))

    result = jax_ls.minimize(
      params,
      data,
      start,
      jax_ls.GaussNewton(n_steps=2, inner=NormalEquationsConfig(n_iter=10)),
    )

    assert result.loss < before
    assert len(result.history) == 2

  def test_bfgs_reduces_the_misfit(self, problem):
    params, _, data = problem
    start = jnp.zeros(params.ls_params.n_dof)
    before = float(jax_ls.near_field_l2_loss(params, data, start))

    result = jax_ls.minimize(
      params, data, start, jax_ls.QuasiNewton(maxiter=10)
    )

    assert result.loss < before


class TestFrequencySweep:
  def test_sweep_reduces_the_error_from_a_zero_start(self):
    """Continuation must make real progress on a hard, high-contrast target.

    Deliberately *not* a head-to-head against inverting the top frequency alone.
    That comparison is real -- the wide-band notebook shows 0.41 against 1.13,
    the latter being worse than the zero guess it started from -- but the margin
    depends strongly on the update budget and the grid size: at a 35-update
    budget it shrinks to 1.2x, which would make a flaky test.  The comparison
    belongs in the notebook, where it runs at full budget and is visible.
    """
    domain = Domain(1.0, 1.0, 32, 32)
    omega_max = 2 * np.pi * (domain.n // 8)
    truth = domain.sample(
      lambda X, Y: (
        3.0
        * (
          jnp.exp(-500 * (jnp.square(X + 0.1) + jnp.square(Y + 0.2)))
          + jnp.exp(-500 * (jnp.square(X - 0.1) + jnp.square(Y - 0.1)))
        )
      )
    )

    def error(nu):
      return float(jnp.linalg.norm(nu - truth) / jnp.linalg.norm(truth))

    swept = jax_ls.frequency_sweep(
      domain,
      {omega_max / 4: 10, omega_max / 2: 10, omega_max: 15},
      lambda params: jax_ls.near_field_map(params, truth),
      radius=1.5,
      verbose=False,
    )

    # error(0) is exactly 1 by construction, so this asserts genuine progress.
    assert error(swept.nu) < 0.75, (
      f"sweep only reached {error(swept.nu):.3f}; a zero guess scores 1.0"
    )

  def test_history_records_one_entry_per_frequency(self):
    domain = Domain(1.0, 1.0, 24, 24)
    truth = domain.sample(
      lambda X, Y: 0.5 * jnp.exp(-200 * (jnp.square(X) + jnp.square(Y)))
    )
    stages = {6.0: 3, 12.0: 3}

    result = jax_ls.frequency_sweep(
      domain,
      stages,
      lambda p: jax_ls.near_field_map(p, truth),
      radius=1.5,
      verbose=False,
    )

    assert [record["omega"] for record in result.history] == [6.0, 12.0]
    # the sampling rule must have been applied, so the counts grow
    assert result.history[1]["n_receivers"] > result.history[0]["n_receivers"]

  def test_band_limiting_is_optional(self):
    """Both settings must run; the projection is an option, not a requirement."""
    domain = Domain(1.0, 1.0, 24, 24)
    truth = domain.sample(
      lambda X, Y: 0.5 * jnp.exp(-200 * (jnp.square(X) + jnp.square(Y)))
    )
    data_for = lambda p: jax_ls.near_field_map(p, truth)  # noqa: E731

    without = jax_ls.frequency_sweep(
      domain, {8.0: 3}, data_for, radius=1.5, band_factor=None, verbose=False
    )
    with_band = jax_ls.frequency_sweep(
      domain, {8.0: 3}, data_for, radius=1.5, band_factor=2.0, verbose=False
    )

    assert np.isfinite(without.loss) and np.isfinite(with_band.loss)


class TestWidebandLoss:
  """The simultaneous loss and its weighting."""

  @pytest.fixture(scope="class")
  def datasets(self):
    domain = Domain(1.0, 1.0, 24, 24)
    truth = domain.sample(
      lambda X, Y: 0.6 * jnp.exp(-200 * (jnp.square(X) + jnp.square(Y)))
    )
    out = []
    for omega in (6.0, 12.0):
      params = jax_ls.make_sweep_params(domain, omega, 1.5)
      out.append(
        jax_ls.FrequencyDataset(
          params=params, data=jax_ls.near_field_map(params, truth)
        )
      )
    return domain, truth, out

  def test_data_norm_makes_every_frequency_contribute_order_one(self, datasets):
    """Each term is a relative misfit, so loss(0) is about K/2."""
    domain, _, sets = datasets
    loss = jax_ls.make_wideband_loss(sets, jax_ls.Weighting.DATA_NORM)

    value = float(loss(jnp.zeros(domain.n_dof)))
    assert value == pytest.approx(len(sets) / 2, rel=0.2)

  def test_uniform_weighting_is_dominated_by_one_frequency(self, datasets):
    """Why DATA_NORM is the default: the raw terms are not comparable."""
    _, _, sets = datasets
    uniform = jax_ls.frequency_weights(sets, jax_ls.Weighting.UNIFORM)
    normalised = jax_ls.frequency_weights(sets, jax_ls.Weighting.DATA_NORM)

    assert float(jnp.max(uniform) / jnp.min(uniform)) == pytest.approx(1.0)
    # the data norms genuinely differ between frequencies
    assert float(jnp.max(normalised) / jnp.min(normalised)) > 1.5

  def test_gradient_matches_finite_differences(self, datasets):
    domain, truth, sets = datasets
    loss = jax.jit(jax_ls.make_wideband_loss(sets))

    start = 0.5 * truth
    direction = domain.sample(
      lambda X, Y: jnp.exp(-300 * (jnp.square(X - 0.1) + jnp.square(Y - 0.1)))
    )
    step = 1e-5

    directional = float(jnp.sum(jax.grad(loss)(start) * direction))
    difference = float(
      (loss(start + step * direction) - loss(start - step * direction))
      / (2 * step)
    )

    err = abs(directional - difference) / abs(difference)
    assert err < 1e-5, f"wideband gradient vs finite differences {err:.2e}"
