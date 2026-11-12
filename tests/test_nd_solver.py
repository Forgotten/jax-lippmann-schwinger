"""Unit tests for the nested-dissection solver with inverted fronts.

`jax_ls.nd_solver` factors any 9-point matrix; most tests here use random
complex ones and compare against SciPy.  The last classes check the solver as
the ``"block_inverse"`` backend of the sparsifying preconditioner: same
preconditioner, same answers, same derivatives as the SuperLU backend.
"""

from functools import partial

import numpy as np
import pytest
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import jax
import jax.numpy as jnp

import jax_ls
from jax_ls import nd_solver as nd
from jax_ls import sparsify
from jax_ls.domain import Domain
from jax_ls.solvers import BlockGmresConfig, GmresConfig

jax.config.update("jax_enable_x64", True)

GRIDS = [(3, 3, 2), (5, 7, 2), (16, 16, 4), (31, 20, 8), (9, 40, 4), (40, 9, 8)]


def random_stencil(m, n, seed=0, diagonal_shift=3.0, dtype=np.complex128):
  """A random 9-point coefficient field and its SciPy matrix."""
  rng = np.random.default_rng(seed)
  coef = rng.standard_normal((9, m, n)) + 1j * rng.standard_normal((9, m, n))
  coef[4] += diagonal_shift
  domain = Domain(1.0, 1.0, n, m)
  coef = (coef * sparsify._valid_mask(domain)).astype(dtype)
  return coef, sparsify._assemble(domain, coef)


def _rel_residual(op, x, b):
  return np.linalg.norm(op @ x - b) / np.linalg.norm(b)


# ---------------------------------------------------------------------------
# The symbolic structure.
# ---------------------------------------------------------------------------
class TestStructure:
  @pytest.mark.parametrize("m,n,leaf", GRIDS)
  def test_separators_partition_the_grid(self, m, n, leaf):
    st = nd.NDStructure.create(m, n, leaf)
    separator = np.concatenate(
      [np.asarray(lv.separator_idx).ravel() for lv in st.levels]
    )
    separator = separator[separator < m * n]
    assert np.array_equal(np.sort(separator), np.arange(m * n))

  @pytest.mark.parametrize("m,n,leaf", GRIDS)
  def test_every_matrix_entry_is_assembled_once(self, m, n, leaf):
    st = nd.NDStructure.create(m, n, leaf)
    _, matrix = random_stencil(m, n)
    sources = np.concatenate([np.asarray(lv.entry_source) for lv in st.levels])
    assert sources.size == matrix.nnz
    assert np.unique(sources).size == matrix.nnz

  def test_entry_targets_are_distinct_within_a_level(self):
    st = nd.NDStructure.create(31, 20, 8)
    for lv in st.levels:
      targets = np.asarray(lv.entry_target)
      assert np.unique(targets).size == targets.size

  def test_boundaries_lie_in_ancestor_separators(self):
    """B of a level is eliminated strictly later: it lies in shallower S."""
    st = nd.NDStructure.create(31, 20, 4)
    later = set()
    for lv in reversed(st.levels):  # root first
      boundary = np.asarray(lv.boundary_idx).ravel()
      boundary = boundary[boundary < st.n_dof]
      assert set(boundary.tolist()) <= later
      later |= set(np.asarray(lv.separator_idx).ravel().tolist())

  def test_root_has_no_boundary(self):
    st = nd.NDStructure.create(16, 16, 4)
    root = st.levels[-1]
    assert root.shape[0] == 1
    assert np.all(np.asarray(root.boundary_idx) == st.n_dof)

  @pytest.mark.parametrize("m,n,leaf", GRIDS)
  def test_nested_dissection_order_matches_the_tree(self, m, n, leaf):
    perm = nd.nested_dissection(m, n, leaf)
    assert np.array_equal(np.sort(perm), np.arange(m * n))
    # Deeper separators come first in the elimination order.
    st = nd.NDStructure.create(m, n, leaf)
    rank = np.empty(m * n, dtype=int)
    rank[perm] = np.arange(m * n)
    for deeper, shallower in zip(st.levels[:-1], st.levels[1:]):
      a = np.asarray(deeper.separator_idx).ravel()
      b = np.asarray(shallower.separator_idx).ravel()
      assert rank[a[a < m * n]].min() < rank[b[b < m * n]].max()

  def test_sparsify_uses_the_same_ordering(self):
    assert np.array_equal(
      sparsify.nested_dissection(31, 20), nd.nested_dissection(31, 20)
    )

  def test_padding_is_modest(self):
    st = nd.NDStructure.create(127, 127, 8)
    exact = 0
    for lv in st.levels:
      separator = np.asarray(lv.separator_idx) < st.n_dof
      boundary = np.asarray(lv.boundary_idx) < st.n_dof
      s, b = separator.sum(1), boundary.sum(1)
      exact += int(np.sum(s * s + 2 * s * b))
    assert st.stored_entries / exact < 1.15

  def test_storage_grows_like_n_log_n(self):
    """Entries per unknown grow slowly, not like sqrt(N)."""
    per_dof = [
      nd.NDStructure.create(n, n, 8).stored_entries / n**2
      for n in (63, 127, 255)
    ]
    assert per_dof[2] / per_dof[0] < 1.5, per_dof

  def test_leaf_must_be_at_least_two(self):
    with pytest.raises(ValueError):
      nd.NDStructure.create(8, 8, 1)


# ---------------------------------------------------------------------------
# Factorisation and solve.
# ---------------------------------------------------------------------------
class TestSolve:
  @pytest.mark.parametrize("m,n,leaf", GRIDS)
  @pytest.mark.parametrize("mode", ["N", "T", "H"])
  def test_matches_the_matrix(self, m, n, leaf, mode):
    coef, matrix = random_stencil(m, n)
    factor = nd.factorize(nd.NDStructure.create(m, n, leaf), jnp.asarray(coef))
    op = {"N": matrix, "T": matrix.T, "H": matrix.conj().T}[mode]
    b = np.random.default_rng(1).standard_normal((m * n, 3)) + 0j
    x = np.asarray(nd.solve(factor, jnp.asarray(b), mode))
    assert _rel_residual(op, x, b) < 1e-12

  def test_agrees_with_scipy(self):
    coef, matrix = random_stencil(20, 24)
    factor = nd.factorize(nd.NDStructure.create(20, 24, 4), jnp.asarray(coef))
    b = np.random.default_rng(2).standard_normal(480) + 1j
    expected = spla.spsolve(matrix.tocsc(), b)
    got = np.asarray(nd.solve(factor, jnp.asarray(b)))
    assert np.linalg.norm(got - expected) <= 1e-11 * np.linalg.norm(expected)

  def test_single_and_block_right_hand_sides_agree(self):
    coef, _ = random_stencil(16, 16)
    factor = nd.factorize(nd.NDStructure.create(16, 16, 4), jnp.asarray(coef))
    rhs_block = jax.random.normal(jax.random.PRNGKey(0), (256, 4)) + 0j
    block = nd.solve(factor, rhs_block)
    columns = jnp.stack(
      [nd.solve(factor, rhs_block[:, j]) for j in range(4)], axis=1
    )
    vmapped = jax.vmap(partial(nd.solve, factor), 1, 1)(rhs_block)
    assert jnp.allclose(block, columns, atol=1e-13)
    assert jnp.allclose(block, vmapped, atol=1e-13)

  def test_factorise_and_solve_jit(self):
    coef, matrix = random_stencil(16, 16)
    st = nd.NDStructure.create(16, 16, 4)
    b = jnp.ones(256, dtype=complex)
    x = jax.jit(lambda c, r: nd.solve(nd.factorize(st, c), r, "T"))(
      jnp.asarray(coef), b
    )
    assert _rel_residual(matrix.T, np.asarray(x), np.asarray(b)) < 1e-12

  def test_flat_coefficients_are_accepted(self):
    coef, matrix = random_stencil(9, 11)
    st = nd.NDStructure.create(9, 11, 4)
    f1 = nd.factorize(st, jnp.asarray(coef))
    f2 = nd.factorize(st, jnp.asarray(coef).reshape((9, -1)))
    b = jnp.ones(99, dtype=complex)
    assert jnp.allclose(nd.solve(f1, b), nd.solve(f2, b))

  def test_single_precision(self):
    coef, matrix = random_stencil(16, 16, dtype=np.complex64)
    factor = nd.factorize(nd.NDStructure.create(16, 16, 4), jnp.asarray(coef))
    assert factor.levels[0].inverse.dtype == jnp.complex64
    b = np.ones(256, dtype=np.complex64)
    x = np.asarray(nd.solve(factor, jnp.asarray(b)))
    assert _rel_residual(matrix.astype(np.complex128), x, b) < 1e-4

  def test_bad_mode_is_rejected(self):
    coef, _ = random_stencil(5, 5)
    factor = nd.factorize(nd.NDStructure.create(5, 5, 2), jnp.asarray(coef))
    with pytest.raises(ValueError):
      nd.solve(factor, jnp.ones(25), "X")


class TestPivoting:
  def test_zero_diagonal_needs_and_gets_pivoting(self):
    """A matrix with a zero diagonal: elimination without pivoting divides by 0.

    ``M = P_shift + 0.1 R``: the identity moved one place to the right in each
    row, plus small random 9-point noise, with the diagonal zeroed.
    """
    m = n = 12
    rng = np.random.default_rng(3)
    coef = 0.1 * (
      rng.standard_normal((9, m, n)) + 1j * rng.standard_normal((9, m, n))
    )
    coef[4] = 0.0  # zero diagonal
    coef[5] += 1.0  # strong (0, +1) neighbour
    coef[3] += 1.0  # and (0, -1), so the last column is not left empty
    domain = Domain(1.0, 1.0, n, m)
    coef = coef * sparsify._valid_mask(domain)
    matrix = sparsify._assemble(domain, coef)
    assert np.all(matrix.diagonal() == 0)
    factor = nd.factorize(nd.NDStructure.create(m, n, 4), jnp.asarray(coef))
    b = np.ones(m * n, dtype=complex)
    x = np.asarray(nd.solve(factor, jnp.asarray(b)))
    assert np.all(np.isfinite(x))
    assert _rel_residual(matrix, x, b) < 1e-10

  def test_diagnostics_are_reported(self):
    coef, _ = random_stencil(16, 16)
    factor = nd.factorize(nd.NDStructure.create(16, 16, 4), jnp.asarray(coef))
    assert 1.0 <= float(factor.growth) < 10.0
    assert 0.0 < float(factor.pivot_ratio) <= 1.0

  def test_stored_inverse_absorbs_the_permutation(self):
    """``Finv`` is the true inverse of the front, pivoting notwithstanding."""
    coef, matrix = random_stencil(5, 5, diagonal_shift=0.0)
    st = nd.NDStructure.create(5, 5, 5)  # one leaf: the whole matrix
    assert len(st.levels) == 1
    factor = nd.factorize(st, jnp.asarray(coef))
    separator = np.asarray(st.levels[0].separator_idx)[0]
    inverse_block = np.asarray(factor.levels[0].inverse)[0]
    dense = matrix.toarray()[np.ix_(separator, separator)]
    assert np.allclose(inverse_block @ dense, np.eye(25), atol=1e-10)


# ---------------------------------------------------------------------------
# As the backend of the sparsifying preconditioner.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def backends():
  n, omega = 40, 30.0
  plain = jax_ls.init_params(1.0, 1.0, n, n, omega)
  superlu = jax_ls.with_sparsifying_preconditioner(plain, backend="superlu")
  block = jax_ls.with_sparsifying_preconditioner(plain, backend="block_inverse")
  nu = plain.domain.sample(lambda X, Y: 2.0 * (X**2 + Y**2 < 0.09) * 1.0)
  f = jax.random.normal(jax.random.PRNGKey(5), (n * n,)) + 0j
  return plain, superlu, block, nu, f


# Restart 200: with 60 the *unpreconditioned* reference stalls at 9e-9 on the
# disc used below, and the comparison would measure that, not the backend.
TIGHT = GmresConfig(tol=1e-12, restart=200, maxiter=20)


def _rel(a, b):
  return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


class TestBackend:
  def test_backend_is_recorded(self, backends):
    _, superlu, block, _, _ = backends
    assert superlu.sparsifier.backend == "superlu"
    assert block.sparsifier.backend == "block_inverse"
    assert superlu.sparsifier.structure is None
    assert isinstance(block.sparsifier.structure, nd.NDStructure)

  def test_auto_picks_superlu_on_cpu(self):
    if jax.default_backend() != "cpu":
      pytest.skip("CPU-only expectation")
    params = jax_ls.init_params(1.0, 1.0, 16, 16, 10.0, sparsify=True)
    assert params.sparsifier.backend == "superlu"

  def test_unknown_backend_is_rejected(self):
    params = jax_ls.init_params(1.0, 1.0, 16, 16, 10.0)
    with pytest.raises(ValueError, match="backend"):
      jax_ls.with_sparsifying_preconditioner(params, backend="pardiso")

  def test_structure_is_shared_across_frequencies(self):
    a = jax_ls.init_params(
      1.0, 1.0, 24, 24, 10.0, sparsify=True, sparsify_backend="block_inverse"
    )
    b = jax_ls.init_params(
      1.0, 1.0, 24, 24, 14.0, sparsify=True, sparsify_backend="block_inverse"
    )
    assert a.sparsifier.structure is b.sparsifier.structure
    assert jax.tree.structure(a) == jax.tree.structure(b)

  def test_same_preconditioner_as_superlu(self, backends):
    _, superlu, block, nu, f = backends
    p_s = jax_ls.preconditioner_for(superlu, nu)(f)
    p_b = jax_ls.preconditioner_for(block, nu)(f)
    assert _rel(p_b, p_s) < 1e-11

  def test_same_adjoint_preconditioner(self, backends):
    _, superlu, block, nu, f = backends
    nu_c = nu * (1 + 0.2j)
    p_s = jax_ls.preconditioner_for(superlu, nu_c, adjoint=True)(f)
    p_b = jax_ls.preconditioner_for(block, nu_c, adjoint=True)(f)
    assert _rel(p_b, p_s) < 1e-11

  def test_factor_matrix_solves_M(self, backends):
    _, _, block, nu, f = backends
    st = block.sparsifier
    stencil, local = st.matrices()
    matrix = -stencil + block.omega**2 * local @ sps.diags(np.asarray(nu))
    factor = sparsify.factor_matrix(st, block.omega, nu)
    x = np.asarray(nd.solve(factor, f))
    assert _rel_residual(matrix, x, np.asarray(f)) < 1e-11
    assert float(factor.growth) < 10.0

  def test_no_host_factorisation(self, backends):
    _, _, block, nu, f = backends
    sparsify.clear_factorizations()
    jax_ls.ls_solver(block, nu, f)
    assert sparsify.factorization_stats().get("factorizations", 0) == 0

  @pytest.mark.parametrize("adjoint", [False, True])
  def test_same_solution(self, backends, adjoint):
    plain, _, block, nu, f = backends
    solver = jax_ls.ls_solver_adj if adjoint else jax_ls.ls_solver
    assert _rel(solver(block, nu, f, TIGHT), solver(plain, nu, f, TIGHT)) < 1e-9

  def test_jvp_and_grad_through_the_solve(self, backends):
    plain, _, block, nu, f = backends
    d = plain.domain.sample(lambda X, Y: jnp.exp(-30 * (X**2 + Y**2)))

    def tangent(params):
      return jax.jvp(
        lambda v: jax_ls.ls_solver(params, v, f, TIGHT), (nu,), (d,)
      )[1]

    def grad(params):
      return jax.grad(
        lambda v: jnp.sum(jnp.abs(jax_ls.ls_solver(params, v, f, TIGHT)) ** 2)
      )(nu)

    assert _rel(tangent(block), tangent(plain)) < 1e-8
    assert _rel(grad(block), grad(plain)) < 1e-8

  def test_vmap_over_sources(self, backends):
    _, superlu, block, nu, f = backends
    rhs_block = jnp.stack([f, 1j * f, jnp.roll(f, 7)], axis=1)

    def solve(params):
      return jax.vmap(lambda b: jax_ls.ls_solver(params, nu, b, TIGHT), 1, 1)(
        rhs_block
      )

    assert _rel(solve(block), solve(superlu)) < 1e-9

  def test_near_field_map_and_block_path(self):
    plain = jax_ls.init_params_near_field(1.0, 1.0, 32, 32, 1.0, 8, 30.0)
    block = plain.replace(
      ls_params=jax_ls.with_sparsifying_preconditioner(
        plain.ls_params, backend="block_inverse"
      )
    )
    nu = plain.ls_params.domain.sample(
      lambda X, Y: jnp.exp(-10 * (X**2 + Y**2))
    )
    d0 = jax_ls.near_field_map(plain, nu)
    assert _rel(jax_ls.near_field_map(block, nu), d0) < 1e-4
    blocked = block.replace(block=BlockGmresConfig(n_steps=8))
    assert _rel(jax_ls.near_field_map(blocked, nu), d0) < 1e-4

  def test_born_pair_is_still_adjoint(self):
    params = jax_ls.init_params_near_field(1.0, 1.0, 32, 32, 1.0, 8, 30.0)
    params = params.replace(
      ls_params=jax_ls.with_sparsifying_preconditioner(
        params.ls_params, backend="block_inverse"
      )
    )
    domain = params.ls_params.domain
    nu = domain.sample(lambda X, Y: jnp.exp(-10 * (X**2 + Y**2)))
    t = domain.sample(lambda X, Y: jnp.exp(-30 * ((X - 0.1) ** 2 + Y**2)))
    w = jax.random.normal(jax.random.PRNGKey(7), (64,)) + 0j
    lin = jax_ls.linearize(params, nu)
    lhs = jnp.vdot(w, lin.jvp(t))
    rhs = jnp.vdot(lin.vjp(w), t)
    assert abs(lhs - rhs) <= 1e-4 * abs(lhs)


# ---------------------------------------------------------------------------
# Caching the factorisation across solves at one contrast.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cached_setup():
  near = jax_ls.init_params_near_field(1.0, 1.0, 48, 48, 1.0, 8, 30.0)
  near = near.replace(
    ls_params=jax_ls.with_sparsifying_preconditioner(
      near.ls_params, backend="block_inverse"
    )
  )
  domain = near.ls_params.domain
  nu = domain.sample(lambda x, y: 2.0 * jnp.exp(-10 * (x**2 + y**2)))
  t = domain.sample(lambda x, y: jnp.exp(-30 * ((x - 0.1) ** 2 + y**2)))
  w = jax.random.normal(jax.random.PRNGKey(9), (64,)) + 0j
  return near, nu, t, w


def _device_factorisations(fn) -> int:
  """Run `fn` once compiled and count the device factorisations it does."""
  jax.block_until_ready(fn())
  sparsify.clear_factorizations()
  jax.block_until_ready(fn())
  return sparsify.factorization_stats().get("device_factorizations", 0)


class TestFactorCaching:
  def test_real_contrast_shares_one_factorisation(self, cached_setup):
    near, nu, _, _ = cached_setup
    factors = jax_ls.preconditioner_factors(near.ls_params, nu)
    assert isinstance(factors, jax_ls.PreconditionerFactors)
    assert factors.adjoint is factors.forward

  def test_complex_contrast_factors_its_conjugate(self, cached_setup):
    near, nu, _, _ = cached_setup
    nu_c = nu * (1.0 + 0.3j)
    factors = jax_ls.preconditioner_factors(near.ls_params, nu_c)
    assert factors.adjoint is not factors.forward
    f = jnp.ones(near.ls_params.n_dof, dtype=complex)
    for solver in (jax_ls.ls_solver, jax_ls.ls_solver_adj):
      cached = solver(near.ls_params, nu_c, f, TIGHT, factors=factors)
      fresh = solver(near.ls_params, nu_c, f, TIGHT)
      assert _rel(cached, fresh) < 1e-12

  def test_nothing_to_cache_without_block_inverse(self):
    params = jax_ls.init_params(1.0, 1.0, 16, 16, 10.0)
    assert jax_ls.preconditioner_factors(params, jnp.zeros(256)) is None
    superlu = jax_ls.with_sparsifying_preconditioner(params, backend="superlu")
    assert jax_ls.preconditioner_factors(superlu, jnp.zeros(256)) is None

  def test_given_factors_are_used_instead_of_refactoring(self, cached_setup):
    near, nu, _, _ = cached_setup
    params = near.ls_params
    factors = jax_ls.preconditioner_factors(params, nu)
    f = jnp.ones(params.n_dof, dtype=complex)
    assert _device_factorisations(lambda: jax_ls.ls_solver(params, nu, f)) == 1
    assert (
      _device_factorisations(
        lambda: jax_ls.ls_solver(params, nu, f, factors=factors)
      )
      == 0
    )

  @pytest.mark.parametrize(
    "name",
    ["map", "jvp_rule", "born_adjoint", "loss_and_grad", "grad", "vjp_rule"],
  )
  def test_each_operation_factors_once(self, cached_setup, name):
    near, nu, t, w = cached_setup
    data = jax_ls.near_field_map(near, 1.1 * nu)
    run = {
      "map": lambda: jax_ls.near_field_map(near, nu),
      "jvp_rule": lambda: jax_ls.born_linearization(near, nu, t),
      "born_adjoint": lambda: jax_ls.born_adjoint(near, nu, w),
      "loss_and_grad": lambda: jax_ls.near_field_l2_loss_and_grad(
        near, data, nu
      ),
      "grad": lambda: jax.grad(partial(jax_ls.near_field_l2_loss, near, data))(
        nu
      ),
      "vjp_rule": lambda: jax.vjp(
        partial(jax_ls.near_field_map_reverse_mode, near), nu
      )[1](w),
    }[name]
    assert _device_factorisations(run) == 1

  @pytest.mark.parametrize("cache", [True, False])
  def test_linearization_factors_once_when_cached(self, cached_setup, cache):
    near, nu, t, _ = cached_setup
    n_pairs = 3

    def run():
      lin = jax_ls.linearize(near, nu, cache_factors=cache)
      return [lin.vjp(lin.jvp(t)) for _ in range(n_pairs)]

    expected = 1 if cache else 1 + 2 * n_pairs
    assert _device_factorisations(run) == expected

  def test_caching_changes_no_result(self, cached_setup):
    near, nu, t, w = cached_setup
    cached = jax_ls.linearize(near, nu)
    fresh = jax_ls.linearize(near, nu, cache_factors=False)
    assert cached.factors is not None and fresh.factors is None
    assert jnp.array_equal(cached.jvp(t), fresh.jvp(t))
    assert jnp.array_equal(cached.vjp(w), fresh.vjp(w))

  def test_gauss_newton_step_factors_once(self, cached_setup):
    """A whole CG-on-normal-equations step reuses the linearisation's factor."""
    near, nu, _, _ = cached_setup
    data = jax_ls.near_field_map(near, 1.1 * nu).reshape(-1)
    inner = jax_ls.NormalEquationsConfig(n_iter=4)
    counts = {}
    for cache in (True, False):

      def run(cache=cache):
        lin = jax_ls.linearize(near, nu, cache_factors=cache)
        return jax_ls.gauss_newton_step(lin, data - lin.data, 30.0, inner)

      counts[cache] = _device_factorisations(run)
    assert counts[True] == 1
    assert counts[False] > 1 + 2 * 4, counts

  def test_superlu_gradient_agrees(self, cached_setup):
    """The cached gradient matches the SuperLU backend's."""
    near, nu, _, _ = cached_setup
    superlu = near.replace(
      ls_params=jax_ls.with_sparsifying_preconditioner(
        near.ls_params.replace(sparsifier=None), backend="superlu"
      )
    )
    data = jax_ls.near_field_map(near, 1.1 * nu)
    _, g_block = jax_ls.near_field_l2_loss_and_grad(near, data, nu)
    _, g_superlu = jax_ls.near_field_l2_loss_and_grad(superlu, data, nu)
    assert _rel(g_block, g_superlu) < 1e-3
