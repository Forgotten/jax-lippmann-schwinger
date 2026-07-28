"""Unit tests for the near_field module.

Tests cover parameter initialisation, the scalar Green's function, the
near-field map, reciprocity, and the L2 loss.  The differentiation entry points
(forward mode, reverse mode, Hermitian adjoint) are covered in
`test_derivatives.py`.
"""

from functools import partial

import numpy as np
import pytest
import scipy as sp

import jax
import jax.numpy as jnp
from jax import jit

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

  def test_g_sample_is_minus_the_green_function(self, params_nf_small):
    """G_sample stores -G; pin the sign convention."""
    params = params_nf_small
    ls = params.ls_params
    omega = float(ls.omega)

    n_theta = params.n_theta
    d_theta = 2 * np.pi / n_theta
    theta = np.linspace(np.pi, 3 * np.pi - d_theta, n_theta)
    x_s = np.cos(theta[0]), np.sin(theta[0])  # radius 1

    r = np.sqrt(
      (np.asarray(ls.X).reshape(-1) - x_s[0]) ** 2
      + (np.asarray(ls.Y).reshape(-1) - x_s[1]) ** 2
    )
    expected = -(1j / 4) * sp.special.hankel1(0, omega * r)
    got = np.asarray(params.G_sample[:, 0])
    err = float(np.max(np.abs(got - expected)) / np.max(np.abs(expected)))
    assert err < 1e-10, f"G_sample sign/scale error {err:.2e}"


class TestInitParamsNearField:
  """Tests for init_params_near_field."""

  def test_g_sample_shape(self, params_nf_small):
    """G_sample should be (n*m, n_theta)."""
    n = 2**6
    assert params_nf_small.G_sample.shape == (n * n, n)
    assert params_nf_small.n_theta == n

  def test_spacing_delegates_to_ls_params(self, params_nf_small):
    """The spacing is stored once, on the Lippmann-Schwinger parameters."""
    n = 2**6
    expected = 1.0 / n  # periodic convention: h = ax / n
    assert float(params_nf_small.hx) == pytest.approx(expected, rel=1e-12)
    assert float(params_nf_small.hy) == pytest.approx(expected, rel=1e-12)
    assert float(params_nf_small.hx) == float(params_nf_small.ls_params.hx)
    assert float(params_nf_small.cell_area) == pytest.approx(
      expected**2, rel=1e-12
    )

  def test_ls_params_embedded(self, params_nf_small):
    """The embedded LippSchwinParams should have correct shapes."""
    n = 2**6
    assert params_nf_small.ls_params.GFFT.shape == (4 * n, 4 * n)
    assert params_nf_small.ls_params.X.shape == (n, n)


class TestNearFieldMap:
  """Tests for near_field_map and the differentiable variants."""

  def test_zero_nu_gives_zero(self, params_nf_small):
    """Near field with nu=0 should be zero."""
    n = 2**6
    nu_vect = jnp.zeros(n * n)
    nf = jax_ls.near_field_map(params_nf_small, nu_vect)
    err = float(jnp.linalg.norm(nf))
    assert err < 1e-6, f"Near field with nu=0 has norm {err:.2e}, expected ~0"

  def test_variants_agree(self, params_nf_small, perturbation_fn):
    """All three entry points must evaluate to the same forward map."""
    params_nf = params_nf_small
    ls = params_nf.ls_params
    nu_vect = jnp.reshape(perturbation_fn(ls.X, ls.Y), (-1,))

    matrix = jax_ls.near_field_map(params_nf, nu_vect).reshape((-1,))
    forward = jax_ls.near_field_map_forward_mode(params_nf, nu_vect)
    reverse = jax_ls.near_field_map_reverse_mode(params_nf, nu_vect)

    for name, other in (("forward_mode", forward), ("reverse_mode", reverse)):
      err = float(jnp.linalg.norm(matrix - other) / jnp.linalg.norm(matrix))
      assert err < 1e-12, f"{name} differs from near_field_map by {err:.2e}"

  def test_reciprocity(self, params_nf_small, perturbation_fn):
    """Sources and receivers coincide, so F must be symmetric.

    The adjoint contraction relies on this; see `_adjoint_apply`.
    """
    params_nf = params_nf_small
    ls = params_nf.ls_params
    nu_vect = jnp.reshape(perturbation_fn(ls.X, ls.Y), (-1,))

    near_field = jax_ls.near_field_map(params_nf, nu_vect)
    err = float(
      jnp.linalg.norm(near_field - near_field.T) / jnp.linalg.norm(near_field)
    )
    assert err < 1e-6, f"Near field is not reciprocal, asymmetry {err:.2e}"

  def test_solver_converged(self, params_nf_small, perturbation_fn):
    """The densities behind the map must actually solve the equation.

    GMRES reports no error when it stagnates, so check the residual directly.
    """
    params_nf = params_nf_small
    ls = params_nf.ls_params
    nu_vect = jnp.reshape(perturbation_fn(ls.X, ls.Y), (-1,))

    rhs = (
      -(float(ls.omega) ** 2) * nu_vect.reshape((-1, 1)) * params_nf.G_sample
    )
    solve = jax.vmap(
      partial(jax_ls.ls_solver, ls, nu_vect), in_axes=1, out_axes=1
    )
    apply = jax.vmap(
      partial(jax_ls.apply_lipp_schwin, ls, nu_vect), in_axes=1, out_axes=1
    )
    residual = apply(solve(rhs)) - rhs

    err = float(
      jnp.max(jnp.linalg.norm(residual, axis=0) / jnp.linalg.norm(rhs, axis=0))
    )
    assert err < 1e-4, f"worst-source GMRES residual {err:.2e} exceeds 1e-4"


class TestNearFieldL2Loss:
  """Tests for near_field_l2_loss and its gradient."""

  GRAD_TOL = 1e-2

  def test_loss_nonnegative(self, params_nf_small, perturbation_fn):
    """L2 loss should be non-negative."""
    params_nf = params_nf_small
    ls = params_nf.ls_params
    nu_vect = jnp.reshape(perturbation_fn(ls.X, ls.Y), (-1,))

    u_data = jax_ls.near_field_map(params_nf, nu_vect)
    assert float(jax_ls.near_field_l2_loss(params_nf, u_data, nu_vect)) >= 0.0

  def test_loss_zero_at_truth(self, params_nf_small, perturbation_fn):
    """Loss at the true nu should be approximately zero."""
    params_nf = params_nf_small
    ls = params_nf.ls_params
    nu_vect = jnp.reshape(perturbation_fn(ls.X, ls.Y), (-1,))

    u_data = jax_ls.near_field_map(params_nf, nu_vect)
    loss = jax_ls.near_field_l2_loss(params_nf, u_data, nu_vect)
    assert float(loss) < 1e-4, f"Loss at truth {float(loss):.2e} should be ~0"

  def test_gradient_vs_fd(
    self, params_nf_small, perturbation_fn, delta_perturbation_fn
  ):
    """Gradient from the custom VJP should match a finite-difference gradient."""
    params_nf = params_nf_small
    ls = params_nf.ls_params

    nu_vect = jnp.reshape(perturbation_fn(ls.X, ls.Y), (-1,))
    delta_nu_vect = jnp.reshape(delta_perturbation_fn(ls.X, ls.Y), (-1,))

    # Data generated from a slightly perturbed nu.
    u_data = jax_ls.near_field_map(params_nf, nu_vect + 0.5 * delta_nu_vect)

    loss_fn = jit(partial(jax_ls.near_field_l2_loss, params_nf, u_data))
    grad_val = jax.grad(loss_fn)(nu_vect)

    delta_t = 1e-3
    fd_directional = float(
      (
        loss_fn(nu_vect + delta_t * delta_nu_vect)
        - loss_fn(nu_vect - delta_t * delta_nu_vect)
      )
      / (2 * delta_t)
    )
    grad_directional = float(jnp.sum(jnp.real(grad_val) * delta_nu_vect))

    err = abs(grad_directional - fd_directional) / (abs(fd_directional) + 1e-12)
    assert err < self.GRAD_TOL, (
      f"Gradient directional derivative error {err:.2e} exceeds "
      f"{self.GRAD_TOL:.0e}"
    )
