"""Unit tests for the near_field module.

Tests cover parameter initialisation, the scalar Green's function,
the near-field map (matrix and vector versions), custom JVP/VJP
consistency, and the L2 loss gradient.
"""

import pytest
import numpy as np
import scipy as sp
import jax
import jax.numpy as jnp
from jax import jit, random
from functools import partial

import jax_ls


class TestGreenFunction:
  """Tests for the scalar Green's function G(r, omega)."""

  def test_matches_hankel(self):
    """green_function should equal (1j/4) * H_0^{(1)}(omega * r)."""
    r = np.linspace(0.1, 5.0, 50)
    omega = 10.0
    g = jax_ls.green_function(r, omega)
    expected = (1j / 4) * sp.special.hankel1(0, omega * r)
    err = float(np.max(np.abs(g - expected)))
    assert err < 1e-10, f"Green's function error {err:.2e} exceeds 1e-10"

  def test_reciprocity(self):
    """G(r, omega) should be symmetric in r (already scalar)."""
    r = 2.5
    omega = 15.0
    assert jax_ls.green_function(r, omega) == pytest.approx(
      jax_ls.green_function(r, omega)
    )


class TestInitParamsNearField:
  """Tests for init_params_near_field."""

  def test_g_sample_shape(self, params_nf_small):
    """G_sample should be (n*m, n_theta)."""
    n = 2**6
    n_theta = n
    assert params_nf_small.G_sample.shape == (n * n, n_theta)

  def test_hx_hy_values(self, params_nf_small):
    """Grid spacing should match ax/(n-1)."""
    n = 2**6
    expected_hx = 1.0 / (n - 1)
    assert float(params_nf_small.hx) == pytest.approx(expected_hx, rel=1e-5)
    assert float(params_nf_small.hy) == pytest.approx(expected_hx, rel=1e-5)

  def test_ls_params_embedded(self, params_nf_small):
    """The embedded LippSchwinParams should have correct shapes."""
    n = 2**6
    assert params_nf_small.ls_params.GFFT.shape == (4 * n, 4 * n)
    assert params_nf_small.ls_params.X.shape == (n, n)


class TestNearFieldMap:
  """Tests for near_field_map and near_field_map_vect."""

  def test_zero_nu_gives_zero(self, params_nf_small):
    """Near field with nu=0 should be zero."""
    n = 2**6
    nu_vect = jnp.zeros(n * n)
    nf = jax_ls.near_field_map_vect(params_nf_small, nu_vect)
    err = float(jnp.linalg.norm(nf))
    assert err < 1e-6, f"Near field with nu=0 has norm {err:.2e}, expected ~0"

  def test_vect_matches_matrix(self, params_nf_small, perturbation_fn):
    """near_field_map_vect (flat) should match near_field_map (matrix)."""
    params_nf = params_nf_small
    n = 2**6

    nu = perturbation_fn(params_nf.ls_params.X, params_nf.ls_params.Y)
    nu_vect = jnp.reshape(nu, (-1,))

    nf_matrix = jax_ls.near_field_map(params_nf, nu_vect)
    nf_vect = jax_ls.near_field_map_vect(params_nf, nu_vect)

    err = float(
      jnp.linalg.norm(nf_matrix.reshape((-1,)) - nf_vect) / jnp.linalg.norm(nf_vect)
    )
    assert err < 1e-6, f"Vect vs matrix near-field error {err:.2e} exceeds 1e-6"


class TestNearFieldJVP:
  """Tests for the custom JVP of near_field_map_vect."""

  BORN_NF_TOL = 1e-1

  def test_jvp_vs_fd(self, params_nf_small, perturbation_fn, delta_perturbation_fn):
    """Custom JVP should approximate centered finite difference."""
    params_nf = params_nf_small

    nu = perturbation_fn(params_nf.ls_params.X, params_nf.ls_params.Y) + 0j
    nu_vect = jnp.reshape(nu, (-1,))

    delta_nu = delta_perturbation_fn(params_nf.ls_params.X, params_nf.ls_params.Y) + 0j
    delta_nu_vect = jnp.reshape(delta_nu, (-1,))

    near_field = jit(partial(jax_ls.near_field_map_vect, params_nf))

    _, J = jax.linearize(near_field, nu_vect)
    nf_born = J(delta_nu_vect)

    delta_t = 0.01
    nf_p = near_field(nu_vect + delta_t * delta_nu_vect)
    nf_m = near_field(nu_vect - delta_t * delta_nu_vect)
    nf_fd = (nf_p - nf_m) / (2 * delta_t)

    err = float(jnp.linalg.norm(nf_born - nf_fd) / jnp.linalg.norm(nf_born))
    assert err < self.BORN_NF_TOL, (
      f"JVP vs FD error {err:.2e} exceeds {self.BORN_NF_TOL:.0e}"
    )


class TestNearFieldVJP:
  """Tests for the custom VJP of near_field_map_vect_vjp."""

  ADJOINT_TOL = 1e-3

  def test_adjoint_consistency(self, params_nf_small, perturbation_fn, rng_key):
    """<J δν, w> ≈ <δν, J* w> for random vectors."""
    params_nf = params_nf_small

    nu = perturbation_fn(params_nf.ls_params.X, params_nf.ls_params.Y)
    nu_vect = jnp.reshape(nu, (-1,))

    near_field_auto = jit(partial(jax_ls.near_field_map_vect, params_nf))
    near_field_vjp_fn = jit(partial(jax_ls.near_field_map_vect_vjp, params_nf))

    y, J = jax.linearize(near_field_auto, nu_vect)
    primals, J_star = jax.vjp(near_field_vjp_fn, nu_vect)

    key1, key2 = random.split(rng_key)
    rand_tgnt = random.normal(key1, shape=(nu_vect.shape[0],))
    rand_cotgnt = random.normal(key2, shape=(y.shape[0],)) + 1j * random.normal(
      key2, shape=(y.shape[0],)
    )

    delta_u = J(rand_tgnt)
    adjoint = J_star(rand_cotgnt)

    lhs = jnp.sum(jnp.conj(rand_cotgnt) * delta_u)
    rhs = jnp.sum(jnp.conj(adjoint[0]) * rand_tgnt)
    err = float(jnp.abs(lhs - rhs) / jnp.abs(lhs))

    assert err < self.ADJOINT_TOL, (
      f"Adjoint test error {err:.2e} exceeds {self.ADJOINT_TOL:.0e}"
    )


class TestNearFieldL2Loss:
  """Tests for near_field_l2_loss and its gradient."""

  GRAD_TOL = 1e-1

  def test_loss_nonnegative(self, params_nf_small, perturbation_fn):
    """L2 loss should be non-negative."""
    params_nf = params_nf_small

    nu = perturbation_fn(params_nf.ls_params.X, params_nf.ls_params.Y)
    nu_vect = jnp.reshape(nu, (-1,))

    # generate synthetic data
    nf_data = jax_ls.near_field_map_vect(params_nf, nu_vect)
    n_angles = params_nf.G_sample.shape[1]
    u_data = nf_data.reshape((n_angles, n_angles))

    loss = jax_ls.near_field_l2_loss(params_nf, u_data, nu_vect)
    assert float(loss) >= 0.0

  def test_loss_zero_at_truth(self, params_nf_small, perturbation_fn):
    """Loss at the true nu should be approximately zero."""
    params_nf = params_nf_small

    nu = perturbation_fn(params_nf.ls_params.X, params_nf.ls_params.Y)
    nu_vect = jnp.reshape(nu, (-1,))

    # use the forward model output as data
    nf_data = jax_ls.near_field_map_vect(params_nf, nu_vect)
    n_angles = params_nf.G_sample.shape[1]
    u_data = nf_data.reshape((n_angles, n_angles))

    loss = jax_ls.near_field_l2_loss(params_nf, u_data, nu_vect)
    assert float(loss) < 1e-4, f"Loss at truth {float(loss):.2e} should be ~0"

  def test_gradient_vs_fd(
    self, params_nf_small, perturbation_fn, delta_perturbation_fn
  ):
    """Gradient from custom VJP should match finite-difference gradient."""
    params_nf = params_nf_small

    nu = perturbation_fn(params_nf.ls_params.X, params_nf.ls_params.Y)
    nu_vect = jnp.reshape(nu, (-1,))

    delta_nu = delta_perturbation_fn(params_nf.ls_params.X, params_nf.ls_params.Y)
    delta_nu_vect = jnp.reshape(delta_nu, (-1,))

    # generate data from a slightly perturbed nu
    nu_data = nu_vect + 0.5 * delta_nu_vect
    nf_data = jax_ls.near_field_map_vect(params_nf, nu_data)
    n_angles = params_nf.G_sample.shape[1]
    u_data = nf_data.reshape((n_angles, n_angles))

    # loss function
    def loss_fn(nu_v):
      return jax_ls.near_field_l2_loss(params_nf, u_data, nu_v)

    # gradient via custom VJP
    grad_val = jax.grad(loss_fn)(nu_vect)

    # directional derivative via FD
    delta_t = 1e-3
    loss_p = loss_fn(nu_vect + delta_t * delta_nu_vect)
    loss_m = loss_fn(nu_vect - delta_t * delta_nu_vect)
    fd_directional = float((loss_p - loss_m) / (2 * delta_t))

    # compare directional derivative
    grad_directional = float(jnp.sum(jnp.real(grad_val) * delta_nu_vect))

    err = abs(grad_directional - fd_directional) / (abs(fd_directional) + 1e-12)
    assert err < self.GRAD_TOL, (
      f"Gradient directional derivative error {err:.2e} exceeds {self.GRAD_TOL:.0e}"
    )
