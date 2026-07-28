"""Unit tests for the core jax_ls module.

Tests cover parameter initialisation, Green's function application, the
Lippmann-Schwinger operator, adjoint relations, and the iterative solver.
Grid conventions and rectangular domains are covered in `test_grid.py`; the
truncated kernel itself in `test_kernel.py`.
"""

from functools import partial

import pytest

import jax
import jax.numpy as jnp
from jax import jit, jvp, random

import jax_ls

jax.config.update("jax_enable_x64", True)


class TestInitParams:
  """Verify shapes and basic properties of init_params."""

  def test_gfft_shape(self, params_small):
    """GFFT should be (pad*m, pad*n)."""
    m, n = params_small.X.shape
    assert params_small.GFFT.shape == (
      jax_ls.DEFAULT_PAD * m,
      jax_ls.DEFAULT_PAD * n,
    )

  def test_grid_shape(self, params_small):
    """X and Y grids should be (m, n)."""
    n = 2**6
    assert params_small.X.shape == (n, n)
    assert params_small.Y.shape == (n, n)

  def test_omega_stored(self, params_small):
    """Omega should be stored as provided."""
    assert float(params_small.omega) == pytest.approx(128.0)

  def test_grid_centered(self, params_small):
    """Grid should be approximately centered at the origin."""
    assert float(jnp.mean(params_small.X)) == pytest.approx(0.0, abs=0.1)
    assert float(jnp.mean(params_small.Y)) == pytest.approx(0.0, abs=0.1)

  def test_n_dof(self, params_small):
    """`n_dof` should be the number of grid points."""
    assert params_small.n_dof == params_small.X.size


class TestGreenFunction:
  """Tests for apply_green_function and apply_conj_green_function."""

  def test_output_shape(self, params_small):
    """Output should have the same shape as input."""
    u = jnp.ones(params_small.n_dof, dtype=complex)
    result = jax_ls.apply_green_function(params_small, u)
    assert result.shape == u.shape

  def test_zero_input(self, params_small):
    """G(0) should be 0."""
    u = jnp.zeros(params_small.n_dof, dtype=complex)
    result = jax_ls.apply_green_function(params_small, u)
    assert float(jnp.linalg.norm(result)) == pytest.approx(0.0, abs=1e-10)

  def test_conj_green_output_shape(self, params_small):
    """Conjugate Green output should match input shape."""
    u = jnp.ones(params_small.n_dof, dtype=complex)
    result = jax_ls.apply_conj_green_function(params_small, u)
    assert result.shape == u.shape

  def test_adjoint_relation(self, params_small, rng_key):
    """<Gu, v> should approximate <u, G*v> for random vectors."""
    size = params_small.n_dof
    key1, key2 = random.split(rng_key)
    u = random.normal(key1, (size,)) + 1j * random.normal(key1, (size,))
    v = random.normal(key2, (size,)) + 1j * random.normal(key2, (size,))

    Gu = jax_ls.apply_green_function(params_small, u)
    Gstar_v = jax_ls.apply_conj_green_function(params_small, v)

    lhs = jnp.sum(jnp.conj(v) * Gu)
    rhs = jnp.sum(jnp.conj(Gstar_v) * u)

    err = float(jnp.abs(lhs - rhs) / jnp.abs(lhs))
    assert err < 1e-4, f"<Gu,v> vs <u,G*v> error {err:.2e} exceeds 1e-4"


class TestLippSchwinOperator:
  """Tests for apply_lipp_schwin and apply_lipp_schwin_adj."""

  def test_zero_nu_returns_neg_u(self, params_small):
    """With nu=0, (-I + omega^2 * 0 * G)u = -u."""
    size = params_small.n_dof
    nu_vect = jnp.zeros(size)
    u = jnp.ones(size, dtype=complex) * (1.0 + 0.5j)

    result = jax_ls.apply_lipp_schwin(params_small, nu_vect, u)
    err = float(jnp.linalg.norm(result + u) / jnp.linalg.norm(u))
    assert err < 1e-6, f"LS with nu=0 error {err:.2e} exceeds 1e-6"

  def test_adjoint_zero_nu_returns_neg_u(self, params_small):
    """With nu=0, adjoint operator also returns -u."""
    size = params_small.n_dof
    nu_vect = jnp.zeros(size)
    u = jnp.ones(size, dtype=complex) * (1.0 + 0.5j)

    result = jax_ls.apply_lipp_schwin_adj(params_small, nu_vect, u)
    err = float(jnp.linalg.norm(result + u) / jnp.linalg.norm(u))
    assert err < 1e-6, f"Adjoint LS with nu=0 error {err:.2e} exceeds 1e-6"

  def test_adjoint_is_a_similarity_transform(self, params_small, rng_key):
    """apply_lipp_schwin_adj = D A^H D^{-1} with D = diag(nu).

    Undoing the similarity transform must recover the true Hermitian adjoint of
    `apply_lipp_schwin`.  The near-field adjoint solve depends on this, so it is
    asserted here directly rather than only end-to-end.
    """
    size = params_small.n_dof
    key1, key2, key3 = random.split(rng_key, 3)
    nu = 1.0 + 0.5 * random.uniform(key1, (size,))  # bounded away from zero
    u = random.normal(key2, (size,)) + 1j * random.normal(key2, (size,))
    v = random.normal(key3, (size,)) + 1j * random.normal(key3, (size,))

    a_u = jax_ls.apply_lipp_schwin(params_small, nu, u)
    # A^H v = D^{-1} (apply_lipp_schwin_adj) D v.
    a_h_v = jax_ls.apply_lipp_schwin_adj(params_small, nu, nu * v) / nu

    lhs = jnp.sum(jnp.conj(v) * a_u)
    rhs = jnp.sum(jnp.conj(a_h_v) * u)
    err = float(jnp.abs(lhs - rhs) / jnp.abs(lhs))
    assert err < 1e-10, f"<Au,v> vs <u,A^H v> error {err:.2e}"


class TestSolver:
  """Tests for ls_solver and related solver functions."""

  def test_solver_zero_nu(self, params_small):
    """With nu=0, A = -I so sigma = -f."""
    size = params_small.n_dof
    nu_vect = jnp.zeros(size)
    f = jnp.ones(size, dtype=complex) * (2.0 + 1.0j)

    sigma = jax_ls.ls_solver(params_small, nu_vect, f)
    err = float(jnp.linalg.norm(sigma + f) / jnp.linalg.norm(f))
    assert err < 1e-5, f"Solver with nu=0 error {err:.2e} exceeds 1e-5"

  def test_solver_residual(self, params_small, perturbation_fn):
    """After solving, ||A sigma - f|| / ||f|| should be small."""
    params = params_small
    omega = float(params.omega)

    nu = perturbation_fn(params.X, params.Y)
    nu_vect = jnp.reshape(nu, (-1,))

    u_i = jnp.exp(1j * omega * params.Y)
    f = jnp.reshape(-(omega**2) * nu * u_i, (-1,))

    sigma = jax_ls.ls_solver(params, nu_vect, f)
    residual = jax_ls.apply_lipp_schwin(params, nu_vect, sigma) - f

    err = float(jnp.linalg.norm(residual) / jnp.linalg.norm(f))
    assert err < 1e-4, f"Solver residual {err:.2e} exceeds 1e-4"

  def test_adjoint_solver_residual(self, params_small, perturbation_fn):
    """The adjoint solve must converge too; GMRES reports no error if not."""
    params = params_small
    omega = float(params.omega)

    nu_vect = jnp.reshape(perturbation_fn(params.X, params.Y), (-1,))
    f = jnp.reshape(
      jnp.exp(1j * omega * params.Y) * perturbation_fn(params.X, params.Y),
      (-1,),
    )

    sigma = jax_ls.ls_solver_adj(params, nu_vect, f)
    residual = jax_ls.apply_lipp_schwin_adj(params, nu_vect, sigma) - f

    err = float(jnp.linalg.norm(residual) / jnp.linalg.norm(f))
    assert err < 1e-4, f"Adjoint solver residual {err:.2e} exceeds 1e-4"

  def test_batched_solver_residual(
    self, params_small, perturbation_fn, rng_key
  ):
    """Every column of a vmap-ed solve must satisfy its own equation."""
    params = params_small
    size = params.n_dof
    nu_vect = jnp.reshape(perturbation_fn(params.X, params.Y), (-1,))

    rhs = random.normal(rng_key, (size, 4)) + 1j * random.normal(
      rng_key, (size, 4)
    )
    solve = jax.vmap(
      partial(jax_ls.ls_solver, params, nu_vect), in_axes=1, out_axes=1
    )
    sigma = solve(rhs)

    apply = jax.vmap(
      partial(jax_ls.apply_lipp_schwin, params, nu_vect), in_axes=1, out_axes=1
    )
    residual = apply(sigma) - rhs

    err = float(
      jnp.max(jnp.linalg.norm(residual, axis=0) / jnp.linalg.norm(rhs, axis=0))
    )
    assert err < 1e-4, f"Batched solver residual {err:.2e} exceeds 1e-4"

  def test_custom_jvp_vs_fd(self, perturbation_fn, delta_perturbation_fn):
    """ls_solver_field custom JVP should approximate centered FD."""
    n = 2**6
    omega = 128.0
    params = jax_ls.init_params(1.0, 1.0, n, n, omega)

    nu = perturbation_fn(params.X, params.Y) + 0j
    nu_vect = jnp.reshape(nu, (-1,))

    delta_nu = delta_perturbation_fn(params.X, params.Y) + 0j
    delta_nu_vect = jnp.reshape(delta_nu, (-1,))

    u_i = jnp.exp(1j * omega * params.Y)
    f = jnp.reshape(-(omega**2) * nu * u_i, (-1,))

    @jit
    def solver_field(nu_v):
      return jax_ls.ls_solver_field(params, nu_v, f)

    delta_t = 0.01
    u_p = solver_field(nu_vect + delta_t * delta_nu_vect)
    u_m = solver_field(nu_vect - delta_t * delta_nu_vect)
    fd_approx = (u_p - u_m) / (2 * delta_t)

    _, u_born = jvp(solver_field, (nu_vect,), (delta_nu_vect,))

    err = float(jnp.linalg.norm(fd_approx - u_born) / jnp.linalg.norm(u_born))
    assert err < 1e-1, f"Custom JVP vs FD error {err:.2e} exceeds 1e-1"
