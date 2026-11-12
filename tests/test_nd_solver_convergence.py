"""Convergence tests for the nested-dissection solver with inverted fronts.

Two things are pinned:

* **The direct solve stays accurate as the grid grows.**  Pivoting is confined
  to each front, so nothing guarantees stability in general; these tests track
  the residual and the element growth of the factorisation on the matrices it
  is actually used for -- sparsifying matrices at ten points per wavelength, up
  to n = 255, smooth and discontinuous, up to contrast 8.
* **As a preconditioner backend it changes nothing about convergence.**  It
  factors the same matrix as SuperLU, exactly, so GMRES must take the same
  number of iterations with either backend across frequency, contrast and
  boundary contact.

Iterations are counted with SciPy's GMRES driving the jitted `jax_ls` operator
and preconditioner, as in `test_sparsify_convergence.py`.
"""

import numpy as np
import pytest
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import jax
import jax.numpy as jnp

import jax_ls
from jax_ls import nd_solver as nd
from jax_ls import sparsify

jax.config.update("jax_enable_x64", True)

RTOL = 1e-6


def _gauss(c: float):
  return lambda x, y: c * jnp.exp(-10 * (x**2 + y**2))


def _disc(c: float, radius: float = 0.3):
  return lambda x, y: c * (x**2 + y**2 < radius**2) * 1.0


def _edge(c: float):
  return lambda x, y: c * (1.0 + 0.5 * jnp.cos(3 * x) * jnp.cos(2 * y))


_PARAMS: dict = {}


def _params(n: int, omega: float, backend: str):
  key = (n, omega, backend)
  if key not in _PARAMS:
    _PARAMS[key] = jax_ls.init_params(
      1.0, 1.0, n, n, omega, sparsify=True, sparsify_backend=backend
    )
  return _PARAMS[key]


def gmres_iterations(params, nu, budget: int = 400) -> tuple[int, float]:
  """GMRES iterations to `RTOL` with the attached preconditioner.

  Args:
    params: Parameters with a sparsifier attached.
    nu: The contrast.
    budget: Maximum iterations, without restarts.

  Returns:
    The iteration count and the true relative residual reached.
  """
  n_dof = params.n_dof
  apply_operator = jax.jit(lambda v: jax_ls.apply_lipp_schwin(params, nu, v))
  apply_precond = jax.jit(jax_ls.preconditioner_for(params, nu))

  def matvec(y: np.ndarray) -> np.ndarray:
    return np.array(apply_operator(apply_precond(jnp.asarray(y))))

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
  x = apply_precond(jnp.asarray(y))
  residual = np.linalg.norm(np.asarray(apply_operator(x)) - b) / np.linalg.norm(
    b
  )
  return count[0], residual


class TestDirectSolveAccuracy:
  @pytest.mark.parametrize("n", [31, 63, 127, 255])
  @pytest.mark.parametrize("profile", ["smooth", "disc"])
  def test_residual_and_growth_as_the_grid_grows(self, n, profile):
    """Ten points per wavelength, contrast 8: residual and growth stay small."""
    omega = 2 * np.pi * n / 10
    params = _params(n, omega, "block_inverse")
    shape = _gauss(8.0) if profile == "smooth" else _disc(8.0)
    nu = params.domain.sample(shape)

    stencil, local = params.sparsifier.matrices()
    matrix = (-stencil + omega**2 * local @ sps.diags(np.asarray(nu))).tocsr()
    factor = sparsify.factor_matrix(params.sparsifier, omega, nu)
    b = np.random.default_rng(n).standard_normal((n * n, 2)) + 0j
    for mode, op in (("N", matrix), ("T", matrix.T)):
      x = np.asarray(nd.solve(factor, jnp.asarray(b), mode))
      residual = np.linalg.norm(op @ x - b) / np.linalg.norm(b)
      assert residual < 1e-10, (mode, residual)
    assert float(factor.growth) < 10.0
    assert float(factor.pivot_ratio) > 1e-6

  def test_leaf_size_does_not_change_the_answer(self):
    params = _params(63, 40.0, "block_inverse")
    nu = params.domain.sample(_disc(4.0))
    coefficients = sparsify.matrix_coefficients(params.sparsifier, 40.0, nu)
    b = jnp.ones(63 * 63, dtype=complex)
    solutions = [
      nd.solve(
        nd.factorize(nd.NDStructure.create(63, 63, leaf), coefficients), b
      )
      for leaf in (2, 4, 8, 16)
    ]
    for x in solutions[1:]:
      assert float(jnp.linalg.norm(x - solutions[0])) < 1e-10 * float(
        jnp.linalg.norm(solutions[0])
      )


class TestSameIterationsAsSuperLU:
  @pytest.mark.parametrize("omega", [20.0, 40.0, 80.0])
  def test_frequency(self, omega):
    counts = {}
    for backend in ("superlu", "block_inverse"):
      params = _params(127, omega, backend)
      its, res = gmres_iterations(params, params.domain.sample(_gauss(2.0)))
      assert res <= 2 * RTOL
      counts[backend] = its
    assert counts["block_inverse"] <= 12, counts
    assert abs(counts["block_inverse"] - counts["superlu"]) <= 1, counts

  @pytest.mark.parametrize(
    "name,shape", [("disc c=4", _disc(4.0)), ("boundary", _edge(1.0))]
  )
  def test_hard_media(self, name, shape):
    counts = {}
    for backend in ("superlu", "block_inverse"):
      params = _params(127, 40.0, backend)
      its, res = gmres_iterations(params, params.domain.sample(shape))
      assert res <= 2 * RTOL
      counts[backend] = its
    assert counts["block_inverse"] <= 25, (name, counts)
    assert abs(counts["block_inverse"] - counts["superlu"]) <= 1, (name, counts)


class TestJaxPath:
  def test_converges_within_one_restart_cycle(self):
    """Fully in JAX -- factorisation, GMRES and preconditioner -- with no
    host factorisation at all."""
    params = _params(127, 80.0, "block_inverse")
    plain = params.replace(sparsifier=None)
    nu = params.domain.sample(_disc(2.0))
    f = jnp.asarray(np.random.default_rng(1).standard_normal(params.n_dof) + 0j)
    config = jax_ls.GmresConfig(tol=1e-6, restart=20, maxiter=2)
    sparsify.clear_factorizations()
    _, info = jax_ls.ls_solver_with_info(params, nu, f, config)
    _, info_plain = jax_ls.ls_solver_with_info(plain, nu, f, config)
    assert bool(info.converged), info
    assert float(info_plain.relative_residual) > 1e-3, info_plain
    assert sparsify.factorization_stats().get("factorizations", 0) == 0

  def test_many_sources_under_vmap(self):
    params = _params(63, 40.0, "block_inverse")
    nu = params.domain.sample(_disc(2.0))
    rng = np.random.default_rng(2)
    rhs = jnp.asarray(rng.standard_normal((params.n_dof, 16)) + 0j)
    solve = jax.jit(
      jax.vmap(lambda b: jax_ls.ls_solver(params, nu, b), in_axes=1, out_axes=1)
    )
    x = solve(rhs)
    residual = (
      jax.vmap(
        lambda v: jax_ls.apply_lipp_schwin(params, nu, v), in_axes=1, out_axes=1
      )(x)
      - rhs
    )
    assert float(jnp.linalg.norm(residual) / jnp.linalg.norm(rhs)) < 1e-4
