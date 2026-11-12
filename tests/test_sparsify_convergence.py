"""Convergence tests for the sparsifying preconditioner.

These pin the *behaviour* Ying's preconditioner is for: GMRES iteration counts
that stay small and nearly flat as the frequency, the contrast and the grid
grow, where unpreconditioned GMRES degrades.  Bounds are set with headroom over
what was measured when this was written, so they guard against regressions
rather than record exact counts:

  n = 127, smooth Gaussian of contrast 2:  omega = 20, 40, 80  ->  ~5-8
  n = 127, omega = 40, contrast 0.5 / 2 / 4                    ->  5 / 7 / 8
  n = 127, omega = 40, disc of contrast 2 / 4                  ->  12 / 15

Iterations are counted exactly with SciPy's GMRES driving the jitted
`jax_ls` operator and preconditioner; the `jax.scipy` solver runs inside a
compiled loop and cannot report a count.  The JAX path itself is checked at the
end, by convergence within a fixed budget.
"""

import numpy as np
import pytest
import scipy.sparse.linalg as spla

import jax
import jax.numpy as jnp

import jax_ls
from jax_ls import sparsify

jax.config.update("jax_enable_x64", True)

RTOL = 1e-6
N = 127


def _gauss(c):
  return lambda X, Y: c * jnp.exp(-10 * (X**2 + Y**2))


def _disc(c, radius=0.3):
  return lambda X, Y: c * (X**2 + Y**2 < radius**2) * 1.0


def _edge(c):
  """A contrast that does *not* vanish at the boundary of the domain."""
  return lambda X, Y: c * (1.0 + 0.5 * jnp.cos(3 * X) * jnp.cos(2 * Y))


_PARAMS = {}


def _params(n, omega):
  if (n, omega) not in _PARAMS:
    _PARAMS[n, omega] = jax_ls.init_params(1.0, 1.0, n, n, omega, sparsify=True)
  return _PARAMS[n, omega]


def gmres_iterations(params, nu, precondition=True, nu_factor=None, budget=400):
  """GMRES iterations to ``RTOL``, and the true relative residual reached.

  Args:
    params: Parameters with a sparsifier attached.
    nu: The contrast of the operator.
    precondition: Whether to right-precondition.
    nu_factor: Build the preconditioner from this contrast instead of `nu`.
    budget: Maximum iterations (no restarts).
  """
  n_dof = params.n_dof
  apply_A = jax.jit(lambda v: jax_ls.apply_lipp_schwin(params, nu, v))
  P = None
  if precondition:
    P = jax.jit(
      jax_ls.preconditioner_for(params, nu if nu_factor is None else nu_factor)
    )

  def matvec(y):
    y = jnp.asarray(y)
    return np.array(apply_A(P(y) if P is not None else y))  # writable copy

  b = np.random.default_rng(0).standard_normal(n_dof) + 0j
  count = [0]
  y, _ = spla.gmres(
    spla.LinearOperator((n_dof, n_dof), matvec=matvec, dtype=complex),
    b,
    rtol=RTOL,
    restart=budget,
    maxiter=1,
    callback=lambda _: count.__setitem__(0, count[0] + 1),
    callback_type="pr_norm",
  )
  x = jnp.asarray(y) if P is None else P(jnp.asarray(y))
  residual = np.linalg.norm(np.asarray(apply_A(x)) - b) / np.linalg.norm(b)
  return count[0], residual


class TestFrequency:
  def test_iterations_stay_flat_as_omega_doubles(self):
    """omega = 20 -> 80 at fixed n: Ying's near frequency-independence."""
    counts = []
    for omega in (20.0, 40.0, 80.0):
      params = _params(N, omega)
      its, res = gmres_iterations(params, params.domain.sample(_gauss(2.0)))
      assert res <= 2 * RTOL
      counts.append(its)
    assert max(counts) <= 12, counts
    assert max(counts) - min(counts) <= 5, counts

  def test_unpreconditioned_gmres_does_not_keep_up(self):
    """The same problem at omega = 80 without it, given 5x the iterations."""
    params = _params(N, 80.0)
    nu = params.domain.sample(_gauss(2.0))
    its, _ = gmres_iterations(params, nu)
    plain, res = gmres_iterations(
      params, nu, precondition=False, budget=5 * its
    )
    assert plain == 5 * its and res > 1e3 * RTOL, (plain, res)


class TestContrast:
  @pytest.mark.parametrize("c,bound", [(0.5, 8), (2.0, 11), (4.0, 12)])
  def test_smooth_contrast(self, c, bound):
    params = _params(N, 40.0)
    its, res = gmres_iterations(params, params.domain.sample(_gauss(c)))
    assert res <= 2 * RTOL and its <= bound, its

  @pytest.mark.parametrize("c,bound", [(2.0, 18), (4.0, 22)])
  def test_discontinuous_contrast(self, c, bound):
    """A jump costs about twice a smooth medium, and no more."""
    params = _params(N, 40.0)
    its, res = gmres_iterations(params, params.domain.sample(_disc(c)))
    assert res <= 2 * RTOL and its <= bound, its

  def test_contrast_reaching_the_boundary(self):
    """Ying assumes ``nu = 0`` near the boundary; the port must not need it.

    This is where the boundary stencils are actually exercised -- and where an
    ordering mismatch between ``Acal`` and ``C`` would show up.
    """
    params = _params(N, 40.0)
    nu = params.domain.sample(_edge(1.0))
    its, res = gmres_iterations(params, nu)
    plain, _ = gmres_iterations(params, nu, precondition=False)
    assert res <= 2 * RTOL and its <= 25 and its < plain / 4, (its, plain)


class TestGrid:
  def test_refinement_at_fixed_frequency(self):
    """More points per wavelength must not cost more iterations."""
    counts = []
    for n in (63, 127):
      params = _params(n, 40.0)
      its, res = gmres_iterations(params, params.domain.sample(_gauss(2.0)))
      assert res <= 2 * RTOL
      counts.append(its)
    assert abs(counts[1] - counts[0]) <= 3, counts

  def test_rectangular_domain(self):
    params = jax_ls.init_params(1.0, 0.5, 96, 48, 40.0, sparsify=True)
    its, res = gmres_iterations(params, params.domain.sample(_gauss(2.0)))
    assert res <= 2 * RTOL and its <= 12, its


class TestStaleFactorisation:
  """What `refactor_rtol` trades: a stale LU costs iterations, not accuracy."""

  @pytest.mark.parametrize("factor,bound", [(0.9, 25), (0.5, 200)])
  def test_scaled_contrast(self, factor, bound):
    params = _params(N, 40.0)
    nu = params.domain.sample(_gauss(2.0))
    exact, _ = gmres_iterations(params, nu)
    stale, res = gmres_iterations(params, nu, nu_factor=factor * nu)
    assert res <= 2 * RTOL
    assert exact < stale <= bound, (exact, stale)


class TestJaxPath:
  def test_converges_within_one_restart_cycle(self):
    """At the default restart of 20 one cycle suffices; unpreconditioned
    GMRES is nowhere near after the same two cycles."""
    params = _params(N, 80.0)
    plain = params.replace(sparsifier=None)
    nu = params.domain.sample(_disc(2.0))
    f = jnp.asarray(np.random.default_rng(1).standard_normal(params.n_dof) + 0j)
    config = jax_ls.GmresConfig(tol=1e-6, restart=20, maxiter=2)
    sparsify.clear_factorizations()
    _, info = jax_ls.ls_solver_with_info(params, nu, f, config)
    _, info_plain = jax_ls.ls_solver_with_info(plain, nu, f, config)
    assert bool(info.converged), info
    assert float(info_plain.relative_residual) > 1e-3, info_plain
    assert sparsify.factorization_stats()["factorizations"] == 1
