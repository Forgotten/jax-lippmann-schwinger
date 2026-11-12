"""Unit tests for Ying's sparsifying preconditioner (`jax_ls.sparsify`).

What is pinned here is *correctness*: that the stencils, the assembled matrices
and the factorisation are what `jax_ls.sparsify` says they are, and that
attaching the preconditioner changes how a solve gets its answer but not the
answer -- nor any derivative built on it.  How *fast* it gets there is in
`test_sparsify_convergence.py`.

The algebra is checked against dense matrices on small grids, where ``G`` can be
formed column by column.
"""

from functools import partial

import numpy as np
import pytest
import scipy.sparse as sps

import jax
import jax.numpy as jnp

import jax_ls
from jax_ls import sparsify
from jax_ls.solvers import BlockGmresConfig, GmresConfig

jax.config.update("jax_enable_x64", True)

N_DENSE = 24
OMEGA_DENSE = 20.0
TIGHT = GmresConfig(tol=1e-11, restart=60, maxiter=20)


def _gauss(c, width=10.0, x0=0.0):
  return lambda X, Y: c * jnp.exp(-width * ((X - x0) ** 2 + Y**2))


def _dense_green(params):
  """``G`` as a dense matrix, one column per kernel application."""
  n_dof = params.n_dof
  apply = jax.jit(jax.vmap(params.kernel.apply))
  return np.asarray(apply(jnp.eye(n_dof, dtype=jnp.complex128))).T


@pytest.fixture(scope="module")
def dense():
  """A small square problem with ``G`` formed densely."""
  params = jax_ls.init_params(1.0, 1.0, N_DENSE, N_DENSE, OMEGA_DENSE)
  params = jax_ls.with_sparsifying_preconditioner(params)
  return params, _dense_green(params)


@pytest.fixture(scope="module")
def rect():
  """A rectangular grid, n != m and ax != ay."""
  params = jax_ls.init_params(1.0, 0.5, 20, 11, 25.0)
  params = jax_ls.with_sparsifying_preconditioner(params)
  return params, _dense_green(params)


@pytest.fixture(scope="module")
def solver_setup():
  """A moderately hard problem for comparing solves with and without."""
  n, omega = 48, 40.0
  plain = jax_ls.init_params(1.0, 1.0, n, n, omega)
  precond = jax_ls.with_sparsifying_preconditioner(plain)
  nu = plain.domain.sample(_gauss(1.5))
  key = jax.random.PRNGKey(3)
  f = jax.random.normal(key, (n * n,)) + 1j * jax.random.normal(
    jax.random.split(key)[0], (n * n,)
  )
  return plain, precond, nu, f


def _rel(a, b):
  return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


# ---------------------------------------------------------------------------
# The premise.
# ---------------------------------------------------------------------------
class TestPremise:
  def test_green_operator_is_complex_symmetric(self, dense):
    """``G^T = G``: the reason ``A^T`` is Ying's field-form operator."""
    _, G = dense
    assert np.linalg.norm(G - G.T) <= 1e-12 * np.linalg.norm(G)

  def test_green_operator_is_toeplitz(self, dense):
    """Translation invariance: why nine stencil types suffice."""
    params, G = dense
    n = params.domain.n
    p, q = 5 * n + 5, 9 * n + 7
    shift = 2 * n + 3
    assert np.isclose(G[p, q], G[p + shift, q + shift], rtol=1e-12)


# ---------------------------------------------------------------------------
# Stencil application.
# ---------------------------------------------------------------------------
class TestStencilApplication:
  @pytest.mark.parametrize("which", ["alpha", "local"])
  def test_apply_matches_assembled_matrix(self, rect, which):
    params, _ = rect
    st = params.sparsifier
    A, C = st.matrices()
    matrix = A if which == "alpha" else C
    v = jax.random.normal(jax.random.PRNGKey(0), (params.n_dof,)) + 0j
    got = sparsify.apply_stencil(st.domain, getattr(st, which), v)
    assert _rel(got, matrix @ np.asarray(v)) < 1e-13

  @pytest.mark.parametrize("which", ["alpha", "local"])
  def test_transpose_matches_assembled_matrix(self, rect, which):
    params, _ = rect
    st = params.sparsifier
    A, C = st.matrices()
    matrix = A if which == "alpha" else C
    w = jax.random.normal(jax.random.PRNGKey(1), (params.n_dof,)) + 0j
    got = sparsify.apply_stencil_transpose(st.domain, getattr(st, which), w)
    assert _rel(got, matrix.T @ np.asarray(w)) < 1e-13

  def test_transpose_is_the_linear_transpose(self, rect):
    """What JAX's own transposition of `apply_stencil` produces."""
    params, _ = rect
    st = params.sparsifier
    v = jnp.zeros(params.n_dof, dtype=jnp.complex128)
    w = jax.random.normal(jax.random.PRNGKey(2), (params.n_dof,)) + 0j
    (auto,) = jax.linear_transpose(
      partial(sparsify.apply_stencil, st.domain, st.alpha), v
    )(w)
    explicit = sparsify.apply_stencil_transpose(st.domain, st.alpha, w)
    assert _rel(explicit, auto) < 1e-13

  def test_stencils_vanish_off_the_grid(self, rect):
    params, _ = rect
    st = params.sparsifier
    invalid = ~sparsify._valid_mask(st.domain)
    assert np.all(np.asarray(st.alpha)[invalid] == 0)
    assert np.all(np.asarray(st.local)[invalid] == 0)

  def test_pattern_is_nine_point(self, rect):
    params, _ = rect
    m, n = params.shape
    A, C = params.sparsifier.matrices()
    expected = (3 * m - 2) * (3 * n - 2)  # neighbours that exist, summed
    assert A.nnz == expected and C.nnz == expected


# ---------------------------------------------------------------------------
# What the stencils do to G.
# ---------------------------------------------------------------------------
class TestSparsification:
  @pytest.mark.parametrize("fixture", ["dense", "rect"])
  def test_local_part_is_exactly_A_G_on_the_pattern(self, fixture, request):
    """``C`` is ``Acal G`` restricted to the 9-point pattern -- at every point.

    This is the test the Julia reference implementation would fail on its
    edges and corners, whose stencil and local blocks were ordered differently.
    """
    params, G = request.getfixturevalue(fixture)
    A, C = params.sparsifier.matrices()
    AG = A @ G
    pattern = (A != 0).toarray()
    assert np.linalg.norm(
      (AG - C.toarray())[pattern]
    ) <= 1e-10 * np.linalg.norm(AG[pattern])

  def test_far_field_is_annihilated(self, dense):
    """Off the pattern ``Acal G`` is small compared with on it."""
    params, G = dense
    A, _ = params.sparsifier.matrices()
    AG = A @ G
    pattern = (A != 0).toarray()
    far = np.linalg.norm(AG[~pattern]) / np.sqrt((~pattern).sum())
    near = np.linalg.norm(AG[pattern]) / np.sqrt(pattern.sum())
    # Without the stencil, the same comparison on G itself is O(1).
    raw_far = np.linalg.norm(G[~pattern]) / np.sqrt((~pattern).sum())
    raw_near = np.linalg.norm(G[pattern]) / np.sqrt(pattern.sum())
    assert far / near < 0.05 * raw_far / raw_near

  def test_quality_is_reported_for_nine_types(self, dense):
    params, _ = dense
    quality = np.asarray(params.sparsifier.quality)
    assert quality.shape == (9,)
    assert np.all((quality > 0) & (quality < 2e-2))

  def test_interior_stencil_is_symmetric(self, dense):
    """A radial kernel gives a stencil with the grid's symmetries.

    Exactly under x <-> y, which maps the far field of the representative point
    onto itself on a square grid.  Only approximately under reflection: on an
    even grid the representative point ``n // 2`` has one more neighbour on one
    side than the other, so its far field is not reflection-symmetric.
    """
    params, _ = dense
    m, n = params.shape
    alpha = np.asarray(params.sparsifier.alpha)[:, m // 2, n // 2].reshape(3, 3)
    alpha = alpha / alpha[1, 1]
    assert np.allclose(alpha, alpha.T, atol=1e-8)
    assert np.allclose(alpha, alpha[::-1, ::-1], rtol=1e-2, atol=0)

  def test_too_small_a_grid_is_rejected(self):
    params = jax_ls.init_params(1.0, 1.0, 2, 8, 5.0)
    with pytest.raises(ValueError, match="3x3"):
      jax_ls.with_sparsifying_preconditioner(params)


# ---------------------------------------------------------------------------
# The factorisation.
# ---------------------------------------------------------------------------
class TestFactorisation:
  @pytest.mark.parametrize("m,n", [(16, 16), (9, 23), (40, 3)])
  def test_nested_dissection_is_a_permutation(self, m, n):
    perm = sparsify.nested_dissection(m, n)
    assert np.array_equal(np.sort(perm), np.arange(m * n))

  @pytest.mark.parametrize("trans", ["N", "T", "H"])
  def test_permuted_lu_solves_the_original_matrix(self, dense, trans):
    params, _ = dense
    A, C = params.sparsifier.matrices()
    nu = np.asarray(params.domain.sample(_gauss(2.0)))
    M = (-A + OMEGA_DENSE**2 * C @ sps.diags(nu)).tocsr()
    lu = sparsify._PermutedLU(M, sparsify.nested_dissection(*params.shape))
    b = np.random.default_rng(0).standard_normal((params.n_dof, 3)) + 0j
    op = {"N": M, "T": M.T, "H": M.conj().T}[trans]
    x = lu.solve(b, trans=trans)
    assert np.linalg.norm(op @ x - b) <= 1e-10 * np.linalg.norm(b)

  def test_preconditioner_is_A_transpose_M_inverse_transpose(self, dense):
    """``P = Acal^T M^{-T}`` exactly, checked densely."""
    params, _ = dense
    A, C = params.sparsifier.matrices()
    nu = params.domain.sample(_gauss(2.0))
    M = (-A + OMEGA_DENSE**2 * C @ sps.diags(np.asarray(nu))).toarray()
    r = jax.random.normal(jax.random.PRNGKey(4), (params.n_dof,)) + 0j
    expected = A.T @ np.linalg.solve(M.T, np.asarray(r))
    got = jax_ls.preconditioner_for(params, nu)(r)
    assert _rel(got, expected) < 1e-10

  def test_adjoint_preconditioner_for_a_complex_contrast(self, dense):
    """``-I + w^2 D_nu G^H = conj(A(conj nu))``, also when nu is complex."""
    params, G = dense
    nu = params.domain.sample(_gauss(1.0)) * (1.0 + 0.3j)
    A_adj = -np.eye(params.n_dof) + OMEGA_DENSE**2 * np.asarray(nu)[:, None] * (
      G.conj()
    )
    A_fwd_conj = np.conj(
      -np.eye(params.n_dof)
      + OMEGA_DENSE**2 * np.conj(np.asarray(nu))[:, None] * G
    )
    assert np.allclose(A_adj, A_fwd_conj)

    r = jax.random.normal(jax.random.PRNGKey(5), (params.n_dof,)) + 0j
    P_adj = jax_ls.preconditioner_for(params, nu, adjoint=True)
    P_fwd = jax_ls.preconditioner_for(params, jnp.conj(nu))
    assert _rel(P_adj(r), jnp.conj(P_fwd(jnp.conj(r)))) < 1e-13

  def test_preconditioner_approximates_the_inverse(self, dense):
    """``A P`` is much closer to the identity than ``-A`` is."""
    params, G = dense
    nu = params.domain.sample(_gauss(2.0))
    A = -np.eye(params.n_dof) + OMEGA_DENSE**2 * np.asarray(nu)[:, None] * G
    P = np.asarray(
      jax.vmap(jax_ls.preconditioner_for(params, nu), 1, 1)(
        jnp.eye(params.n_dof, dtype=jnp.complex128)
      )
    )
    eig = np.linalg.eigvals(A @ P)
    eig_plain = np.linalg.eigvals(-A)
    spread = np.max(np.abs(eig - 1))
    spread_plain = np.max(np.abs(eig_plain - 1))
    assert spread < 0.5 * spread_plain


class TestCache:
  def setup_method(self):
    sparsify.clear_factorizations()

  def test_one_factorisation_serves_every_solve_at_one_contrast(
    self, solver_setup
  ):
    _, params, nu, f = solver_setup
    F = jnp.stack([f, 1j * f, f.conj()], axis=1)
    jax.vmap(lambda b: jax_ls.ls_solver(params, nu, b), 1, 1)(F)
    jax_ls.ls_solver_adj(params, nu, f)
    stats = sparsify.factorization_stats()
    assert stats["factorizations"] == 1
    assert stats["hits"] > 3

  def test_a_new_contrast_is_refactored(self, solver_setup):
    _, params, nu, f = solver_setup
    jax_ls.ls_solver(params, nu, f)
    jax_ls.ls_solver(params, 1.1 * nu, f)
    assert sparsify.factorization_stats()["factorizations"] == 2

  def test_refactor_tolerance_reuses_a_nearby_factorisation(self, solver_setup):
    plain, _, nu, f = solver_setup
    params = jax_ls.with_sparsifying_preconditioner(plain, refactor_rtol=0.2)
    jax_ls.ls_solver(params, nu, f)
    x, info = jax_ls.ls_solver_with_info(params, 1.1 * nu, f, TIGHT)
    stats = sparsify.factorization_stats()
    assert stats["factorizations"] == 1 and stats["reuses"] >= 1
    # A stale factorisation costs iterations, never accuracy.
    assert bool(info.converged)

  def test_cache_size_is_bounded(self, solver_setup):
    _, params, nu, f = solver_setup
    sparsify.set_cache_size(2)
    try:
      for scale in (1.0, 1.1, 1.2):
        jax_ls.ls_solver(params, scale * nu, f)
      assert len(sparsify._REGISTRY.factors) == 2
    finally:
      sparsify.set_cache_size(sparsify.DEFAULT_CACHE_SIZE)

  def test_cache_size_must_be_positive(self):
    with pytest.raises(ValueError):
      sparsify.set_cache_size(0)


# ---------------------------------------------------------------------------
# Plumbing: nothing changes unless asked, and frequencies do not recompile.
# ---------------------------------------------------------------------------
class TestPlumbing:
  def test_default_has_no_preconditioner(self):
    params = jax_ls.init_params(1.0, 1.0, 16, 16, 10.0)
    assert params.sparsifier is None
    assert jax_ls.preconditioner_for(params, jnp.zeros(256)) is None

  def test_init_params_flag(self):
    params = jax_ls.init_params(1.0, 1.0, 16, 16, 10.0, sparsify=True)
    assert isinstance(params.sparsifier, jax_ls.SparsifyingStencils)

  def test_changing_frequency_keeps_the_pytree_structure(self):
    """The host handle is a leaf, so a sweep reuses compiled code."""
    a = jax_ls.init_params(1.0, 1.0, 16, 16, 10.0, sparsify=True)
    b = jax_ls.init_params(1.0, 1.0, 16, 16, 14.0, sparsify=True)
    assert jax.tree.structure(a) == jax.tree.structure(b)
    assert int(a.sparsifier.key) != int(b.sparsifier.key)

  def test_sweep_params_forward_the_flag(self):
    domain = jax_ls.Domain(1.0, 1.0, 16, 16)
    params = jax_ls.make_sweep_params(domain, 10.0, 1.0, sparsify=True)
    assert params.ls_params.sparsifier is not None


# ---------------------------------------------------------------------------
# Same answers, with and without.
# ---------------------------------------------------------------------------
class TestSameAnswers:
  def test_forward_solve(self, solver_setup):
    plain, precond, nu, f = solver_setup
    x0 = jax_ls.ls_solver(plain, nu, f, TIGHT)
    x1, info = jax_ls.ls_solver_with_info(precond, nu, f, TIGHT)
    assert bool(info.converged)
    assert _rel(x1, x0) < 1e-9

  def test_adjoint_solve(self, solver_setup):
    plain, precond, nu, f = solver_setup
    x0 = jax_ls.ls_solver_adj(plain, nu, f, TIGHT)
    x1, info = jax_ls.ls_solver_adj_with_info(precond, nu, f, TIGHT)
    assert bool(info.converged)
    assert _rel(x1, x0) < 1e-9

  def test_warm_start_enters_as_a_correction(self, solver_setup):
    _, precond, nu, f = solver_setup
    exact = jax_ls.ls_solver(precond, nu, f, TIGHT)
    guess = exact + 1e-3 * f
    warm = jax_ls.ls_solver(precond, nu, f, TIGHT, x0=guess)
    assert _rel(warm, exact) < 1e-9

  def test_tolerance_is_on_the_true_residual(self, solver_setup):
    """Right preconditioning: what GMRES stops on is what `SolveInfo` measures."""
    _, precond, nu, f = solver_setup
    for tol in (1e-4, 1e-7):
      _, info = jax_ls.ls_solver_with_info(
        precond, nu, f, GmresConfig(tol=tol, solve_method="incremental")
      )
      assert float(info.relative_residual) <= tol

  def test_jvp_through_the_solve(self, solver_setup):
    """`custom_linear_solve` differentiates the preconditioner consistently."""
    plain, precond, nu, f = solver_setup
    d = plain.domain.sample(_gauss(1.0, width=30.0))

    def tangent(params):
      return jax.jvp(
        lambda v: jax_ls.ls_solver(params, v, f, TIGHT), (nu,), (d,)
      )[1]

    assert _rel(tangent(precond), tangent(plain)) < 1e-8

  def test_grad_through_the_solve(self, solver_setup):
    plain, precond, nu, f = solver_setup

    def grad(params):
      return jax.grad(
        lambda v: jnp.sum(jnp.abs(jax_ls.ls_solver(params, v, f, TIGHT)) ** 2)
      )(nu)

    assert _rel(grad(precond), grad(plain)) < 1e-8

  def test_field_solver_jvp(self, solver_setup):
    plain, precond, nu, f = solver_setup
    d = plain.domain.sample(_gauss(1.0, width=30.0))
    t0 = jax.jvp(partial(jax_ls.ls_solver_field, plain, f=f), (nu,), (d,))[1]
    t1 = jax.jvp(partial(jax_ls.ls_solver_field, precond, f=f), (nu,), (d,))[1]
    assert _rel(t1, t0) < 1e-4  # default tolerance inside ls_solver_field


@pytest.fixture(scope="module")
def near_field_setup():
  plain = jax_ls.init_params_near_field(1.0, 1.0, 32, 32, 1.0, 8, 30.0)
  precond = plain.replace(
    ls_params=jax_ls.with_sparsifying_preconditioner(plain.ls_params)
  )
  nu = plain.ls_params.domain.sample(_gauss(1.0))
  return plain, precond, nu


class TestNearField:
  def test_map(self, near_field_setup):
    plain, precond, nu = near_field_setup
    d0 = jax_ls.near_field_map(plain, nu)
    d1 = jax_ls.near_field_map(precond, nu)
    assert _rel(d1, d0) < 1e-4

  def test_born_pair_is_still_adjoint(self, near_field_setup):
    """``<w, J t> = <J^H w, t>`` with both sides preconditioned."""
    _, precond, nu = near_field_setup
    t = precond.ls_params.domain.sample(_gauss(1.0, width=30.0, x0=0.1))
    key = jax.random.PRNGKey(7)
    w = jax.random.normal(key, (precond.n_sources * precond.n_receivers,)) + 0j
    lin = jax_ls.linearize(precond, nu)
    lhs = jnp.vdot(w, lin.jvp(t))
    rhs = jnp.vdot(lin.vjp(w), t)
    assert abs(lhs - rhs) <= 1e-4 * abs(lhs)

  def test_loss_gradient(self, near_field_setup):
    plain, precond, nu = near_field_setup
    data = jax_ls.near_field_map(plain, 1.2 * nu)
    _, g0 = jax_ls.near_field_l2_loss_and_grad(plain, data, nu)
    _, g1 = jax_ls.near_field_l2_loss_and_grad(precond, data, nu)
    assert _rel(g1, g0) < 1e-3

  def test_block_path(self, near_field_setup):
    plain, precond, nu = near_field_setup
    block = precond.replace(block=BlockGmresConfig(n_steps=8))
    d0 = jax_ls.near_field_map(plain, nu)
    d1 = jax_ls.near_field_map(block, nu)
    assert _rel(d1, d0) < 1e-4
