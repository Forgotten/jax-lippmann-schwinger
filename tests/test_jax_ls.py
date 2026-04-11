"""Unit tests for the core jax_ls module.

Tests cover parameter initialisation, Green's function application,
Lippmann-Schwinger operator, adjoint relations, and the iterative solver.
"""

import jax_ls
from functools import partial
from jax import jit, jvp, random
import jax.numpy as jnp
import pytest
import jax

jax.config.update("jax_enable_x64", True)


class TestInitParams:
  """Verify shapes and basic properties of init_params."""

  def test_gfft_shape(self, params_small):
    """GFFT should be (4n, 4m)."""
    n = 2**6
    assert params_small.GFFT.shape == (4 * n, 4 * n)

  def test_grid_shape(self, params_small):
    """X and Y grids should be (n, m)."""
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


class TestGreenFunction:
  """Tests for apply_green_function and apply_conj_green_function."""

  def test_output_shape(self, params_small):
    """Output should have the same shape as input."""
    n = 2**6
    u = jnp.ones(n * n, dtype=complex)
    result = jax_ls.apply_green_function(params_small, u)
    assert result.shape == u.shape

  def test_zero_input(self, params_small):
    """G(0) should be 0."""
    n = 2**6
    u = jnp.zeros(n * n, dtype=complex)
    result = jax_ls.apply_green_function(params_small, u)
    assert float(jnp.linalg.norm(result)) == pytest.approx(0.0, abs=1e-10)

  def test_conj_green_output_shape(self, params_small):
    """Conjugate Green output should match input shape."""
    n = 2**6
    u = jnp.ones(n * n, dtype=complex)
    result = jax_ls.apply_conj_green_function(params_small, u)
    assert result.shape == u.shape

  def test_adjoint_relation(self, params_small, rng_key):
    """<Gu, v> should approximate <u, G*v> for random vectors."""
    n = 2**6
    key1, key2 = random.split(rng_key)
    u = random.normal(key1, (n * n,)) + 1j * random.normal(key1, (n * n,))
    v = random.normal(key2, (n * n,)) + 1j * random.normal(key2, (n * n,))

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
    n = 2**6
    nu_vect = jnp.zeros(n * n)
    u = jnp.ones(n * n, dtype=complex) * (1.0 + 0.5j)

    result = jax_ls.apply_lipp_schwin(params_small, nu_vect, u)
    err = float(jnp.linalg.norm(result + u) / jnp.linalg.norm(u))
    assert err < 1e-6, f"LS with nu=0 error {err:.2e} exceeds 1e-6"

  def test_adjoint_zero_nu_returns_neg_u(self, params_small):
    """With nu=0, adjoint operator also returns -u."""
    n = 2**6
    nu_vect = jnp.zeros(n * n)
    u = jnp.ones(n * n, dtype=complex) * (1.0 + 0.5j)

    result = jax_ls.apply_lipp_schwin_adj(params_small, nu_vect, u)
    err = float(jnp.linalg.norm(result + u) / jnp.linalg.norm(u))
    assert err < 1e-6, f"Adjoint LS with nu=0 error {err:.2e} exceeds 1e-6"


class TestSolver:
  """Tests for ls_solver and related solver functions."""

  def test_solver_zero_nu(self, params_small):
    """With nu=0, A = -I so sigma = -f."""
    n = 2**6
    nu_vect = jnp.zeros(n * n)
    f = jnp.ones(n * n, dtype=complex) * (2.0 + 1.0j)

    sigma = jax_ls.ls_solver(params_small, nu_vect, f)
    err = float(jnp.linalg.norm(sigma + f) / jnp.linalg.norm(f))
    assert err < 1e-5, f"Solver with nu=0 error {err:.2e} exceeds 1e-5"

  def test_solver_residual(self, params_small, perturbation_fn):
    """After solving, ||A sigma - f|| / ||f|| should be small."""
    params = params_small
    n = 2**6
    omega = float(params.omega)

    nu = perturbation_fn(params.X, params.Y)
    nu_vect = jnp.reshape(nu, (-1,))

    s = jnp.array([0.0, 1.0])
    u_i = jnp.exp(1j * omega * (params.X * s[0] + params.Y * s[1]))
    rhs = -(omega**2) * nu * u_i
    f = jnp.reshape(rhs, (-1,))

    sigma = jax_ls.ls_solver(params, nu_vect, f)
    residual = jax_ls.apply_lipp_schwin(params, nu_vect, sigma) - f

    err = float(jnp.linalg.norm(residual) / jnp.linalg.norm(f))
    assert err < 1e-4, f"Solver residual {err:.2e} exceeds 1e-4"

  def test_custom_jvp_vs_fd(self, perturbation_fn, delta_perturbation_fn):
    """ls_solver_u custom JVP should approximate centered FD."""
    n = 2**6
    omega = 128.0
    params = jax_ls.init_params(1.0, 1.0, n, n, omega)

    nu = perturbation_fn(params.X, params.Y) + 0j
    nu_vect = jnp.reshape(nu, (-1,))

    delta_nu = delta_perturbation_fn(params.X, params.Y) + 0j
    delta_nu_vect = jnp.reshape(delta_nu, (-1,))

    s = jnp.array([0.0, 1.0])
    u_i = jnp.exp(1j * omega * (params.X * s[0] + params.Y * s[1]))
    rhs = -(omega**2) * nu * u_i
    f = jnp.reshape(rhs, (-1,))

    @jit
    def solver_u(nu_v):
      return jax_ls.ls_solver_u(params, nu_v, f)

    delta_t = 0.01
    u_p = solver_u(nu_vect + delta_t * delta_nu_vect)
    u_m = solver_u(nu_vect - delta_t * delta_nu_vect)
    fd_approx = (u_p - u_m) / (2 * delta_t)

    _, u_born = jvp(solver_u, (nu_vect,), (delta_nu_vect,))

    err = float(jnp.linalg.norm(fd_approx - u_born) / jnp.linalg.norm(u_born))
    assert err < 1e-1, f"Custom JVP vs FD error {err:.2e} exceeds 1e-1"
