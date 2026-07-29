"""Unit tests for `GmresConfig`, `SolveInfo` and the configurable solvers.

Covers control and visibility: that the settings take effect, and that a solve
which does *not* converge says so.

Deliberately not covered here, because it already is elsewhere: that the
solvers solve at all -- `test_jax_ls.py::TestSolver` and
`test_grid.py::TestRectangularDomain::test_solver_residual` assert the residual
of a default solve.
"""

import pytest

import jax
import jax.numpy as jnp

import jax_ls
from jax_ls.solvers import GmresConfig, solve, solve_with_info

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def system():
  """A moderately hard solve: contrast 4, where GMRES is known to struggle."""
  params = jax_ls.init_params(1.0, 1.0, 32, 32, 30.0)
  contrast = params.domain.sample(
    lambda X, Y: 4.0 * jnp.exp(-100 * (jnp.square(X) + jnp.square(Y)))
  )
  rhs = jnp.ones(params.n_dof, dtype=complex)
  return params, contrast, rhs


class TestGmresConfig:
  def test_is_all_static(self):
    """Settings are configuration, not data, so they carry no pytree leaves."""
    assert jax.tree.leaves(GmresConfig()) == []

  def test_defaults_match_the_previous_hardcoded_behaviour(self):
    """The defaults must not silently change any existing result."""
    config = GmresConfig()
    assert (config.tol, config.atol, config.restart, config.maxiter) == (
      1e-5,
      0.0,
      20,
      None,
    )
    assert config.solve_method == "batched"


class TestSolveInfo:
  def test_reports_a_starved_solve_as_unconverged(self, system):
    """The capability that motivated the module.

    GMRES's own `info` is always 0, so this is the only honest signal; a solve
    cut off at two restarts is badly wrong and must not pass silently.
    """
    params, contrast, rhs = system
    _, info = jax_ls.ls_solver_with_info(
      params, contrast, rhs, GmresConfig(restart=5, maxiter=2)
    )

    assert not bool(info.converged)
    assert float(info.relative_residual) > 1e-3
    assert "NOT CONVERGED" in str(info)

  def test_reports_a_healthy_solve_as_converged(self, system):
    params, contrast, rhs = system
    _, info = jax_ls.ls_solver_with_info(params, contrast, rhs)

    assert bool(info.converged)
    assert float(info.relative_residual) < GmresConfig().tol

  def test_residual_matches_a_manual_computation(self, system):
    """`relative_residual` must be the true residual, not a proxy."""
    params, contrast, rhs = system
    sigma, info = jax_ls.ls_solver_with_info(params, contrast, rhs)

    expected = jnp.linalg.norm(
      jax_ls.apply_lipp_schwin(params, contrast, sigma) - rhs
    ) / jnp.linalg.norm(rhs)

    assert float(info.relative_residual) == pytest.approx(
      float(expected), rel=1e-10
    )

  def test_zero_right_hand_side_does_not_divide_by_zero(self, system):
    """A zero RHS has a zero solution; that is convergence, not a nan."""
    params, contrast, _ = system
    _, info = jax_ls.ls_solver_with_info(
      params, contrast, jnp.zeros(params.n_dof, dtype=complex)
    )

    assert bool(jnp.isfinite(info.relative_residual))
    assert bool(info.converged)


class TestConfigurableSolvers:
  def test_restart_and_maxiter_take_effect(self, system):
    """Starving the Krylov space must visibly degrade the answer."""
    params, contrast, rhs = system

    generous = jax_ls.ls_solver(params, contrast, rhs)
    starved = jax_ls.ls_solver(
      params, contrast, rhs, GmresConfig(restart=5, maxiter=2)
    )

    def residual(sigma):
      return float(
        jnp.linalg.norm(jax_ls.apply_lipp_schwin(params, contrast, sigma) - rhs)
        / jnp.linalg.norm(rhs)
      )

    assert residual(starved) > 100 * residual(generous)

  def test_warm_start_reaches_the_same_solution(self, system):
    """`x0` is what a frequency sweep will use; it must not change the answer."""
    params, contrast, rhs = system

    cold = jax_ls.ls_solver(params, contrast, rhs)
    warm = jax_ls.ls_solver(params, contrast, rhs, x0=cold)

    assert float(jnp.linalg.norm(warm - cold) / jnp.linalg.norm(cold)) < 1e-10

  def test_reporting_variant_returns_the_same_solution(self, system):
    """`solve_with_info` must only observe, never alter."""
    params, contrast, rhs = system
    operator = lambda x: jax_ls.apply_lipp_schwin(params, contrast, x)  # noqa: E731

    plain = solve(operator, rhs, GmresConfig())
    reported, _ = solve_with_info(operator, rhs, GmresConfig())

    assert bool(jnp.array_equal(plain, reported))

  def test_adjoint_solver_is_configurable_too(self, system):
    """The adjoint path must not be the one that stays hardcoded."""
    params, contrast, rhs = system
    _, info = jax_ls.ls_solver_adj_with_info(
      params, contrast, rhs, GmresConfig(restart=5, maxiter=2)
    )

    assert not bool(info.converged)
