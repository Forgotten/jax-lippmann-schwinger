"""Integration tests: Fourier-accelerated Green's function vs. references.

1. Compares the LS solver at two resolutions (Fourier vs direct quadrature).
2. Compares the FFT-based volume potential G*f against direct quadrature
   using the known Green's function G(r) = (i/4) H_0^{(1)}(omega r),
   following the methodology of Vico, Greengard & Ferrando (2016),
   arXiv:1604.03155.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import jax_ls

jax.config.update("jax_enable_x64", True)


class TestIntegration:
  """Integration tests for the LS solver at multiple resolutions."""

  def test_fourier_vs_quadrature(
    self, params_small, params_double, perturbation_fn
  ):
    """Fourier Green's function application should match direct quadrature."""
    params = params_small
    params_2 = params_double

    m, n = params.X.shape
    omega = float(params.omega)

    # --- solve on the smaller domain ---
    nu = perturbation_fn(params.X, params.Y)
    nu_vect = jnp.reshape(nu, (-1,))

    u_i = jnp.exp(1j * omega * params.Y)
    f = jnp.reshape(-(omega**2) * nu * u_i, (-1,))

    sigma = jax_ls.ls_solver(params, nu_vect, f)

    # --- solve on the larger domain ---
    nu_2 = perturbation_fn(params_2.X, params_2.Y)
    nu_vect_2 = jnp.reshape(nu_2, (-1,))

    u_2_i = jnp.exp(1j * omega * params_2.Y)
    f_2 = jnp.reshape(-(omega**2) * nu_2 * u_2_i, (-1,))

    sigma_2 = jax_ls.ls_solver(params_2, nu_vect_2, f_2)
    u_2_s = jax_ls.apply_green_function(params_2, sigma_2)

    # --- compare at evaluation points ---
    x_eval = params_2.X[-10, :]
    y_eval = params_2.Y[-10, :]

    R = jnp.sqrt(
      jnp.square(params.X.reshape((1, -1)) - x_eval.reshape(-1, 1))
      + jnp.square(params.Y.reshape((1, -1)) - y_eval.reshape(-1, 1))
    )
    U_i = -jax_ls.green_function(R, omega)
    hx = float(params.hx)
    res = -params.cell_area * U_i @ sigma
    ref = u_2_s.reshape(params_2.X.shape)[-10, :]

    err = float(jnp.sqrt(jnp.sum(jnp.square(jnp.abs(res - ref)) * hx)))

    # Dynamic tolerance based on grid element size.
    tol = 0.005 + hx * 0.5
    assert err < tol, (
      f"Fourier vs quadrature error {err:.2e} exceeds dynamic tolerance "
      f"{tol:.2e} (based on hx={hx:.4f})"
    )


class TestAnalyticalGreenConvolution:
  """Compare FFT-based volume potential G*f against direct quadrature
  with the known Green's function G(r) = (i/4) H_0^{(1)}(omega r).

  This verifies the fast convolution from Vico, Greengard & Ferrando
  (arXiv:1604.03155) reproduces the expected result.

  Strategy:
    - Choose a smooth, well-localised Gaussian source f.
    - Compute u = G*f via the FFT method (apply_green_function).
    - At a set of exterior evaluation points (away from the source
      support), compute u_ref by direct quadrature using the
      analytically known Green's function.
    - Compare u and u_ref.
  """

  def test_volume_potential_vs_direct_quadrature(
    self, direct_volume_potential, nearest_grid_index
  ):
    """FFT Green's convolution should match direct Hankel quadrature.

    Evaluation points are taken on a circle of radius 0.35, outside the
    effective support of the Gaussian source, and snapped to the grid so that
    the FFT value and the quadrature refer to the same location.
    """
    n = 2**7
    omega = 16.0
    params = jax_ls.init_params(1.0, 1.0, n, n, omega)

    # Smooth, well-localised Gaussian source.
    f = jnp.exp(-200.0 * (jnp.square(params.X) + jnp.square(params.Y)))
    f_vect = jnp.reshape(f, (-1,)) + 0j

    u_fft = jax_ls.apply_green_function(params, f_vect).reshape((n, n))

    n_eval = 16
    theta_eval = np.linspace(0, 2 * np.pi, n_eval, endpoint=False)
    r_eval = 0.35

    errors = []
    for theta in theta_eval:
      ix, iy = nearest_grid_index(
        params, r_eval * np.cos(theta), r_eval * np.sin(theta)
      )
      reference = direct_volume_potential(params, f_vect, ix, iy)
      errors.append(abs(complex(u_fft[iy, ix]) - reference) / abs(reference))

    err = float(np.max(errors))
    assert err < 1e-9, f"Volume potential error {err:.2e} exceeds 1e-9"

  @pytest.mark.parametrize("n", [2**6, 2**7, 2**8])
  def test_volume_potential_at_several_resolutions(
    self, n, direct_volume_potential, nearest_grid_index
  ):
    """Agreement with direct quadrature must hold at every resolution.

    Under the periodic grid convention the FFT convolution reproduces the
    quadrature sum to round-off, so this is a much sharper statement than
    checking that a discretisation error decreases under refinement.
    """
    omega = 16.0
    params = jax_ls.init_params(1.0, 1.0, n, n, omega)

    f = jnp.exp(-200.0 * (jnp.square(params.X) + jnp.square(params.Y)))
    f_vect = jnp.reshape(f, (-1,)) + 0j

    u_fft = jax_ls.apply_green_function(params, f_vect).reshape((n, n))

    ix, iy = nearest_grid_index(params, 0.3, 0.2)
    reference = direct_volume_potential(params, f_vect, ix, iy)
    err = abs(complex(u_fft[iy, ix]) - reference) / abs(reference)

    assert err < 1e-9, f"n={n}: error {err:.2e} exceeds 1e-9"
