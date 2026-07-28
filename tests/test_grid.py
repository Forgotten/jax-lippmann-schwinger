"""Tests for the grid conventions and for rectangular domains.

The domain is treated as periodic: `n` points of spacing `ax / n` covering
`[-ax/2, ax/2)`, with no duplicated endpoint.  That is the grid the FFT-based
convolution assumes, so the stored quadrature weight `hx * hy` has to agree with
it -- these tests pin that agreement, and that `ax != ay` / `n != m` work.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import jax_ls

jax.config.update("jax_enable_x64", True)


class TestGridConventions:
  """The periodic grid convention: h = a / n, half-open interval."""

  @pytest.mark.parametrize("ax,n", [(1.0, 2**6), (2.0, 2**5), (0.5, 2**7)])
  def test_spacing_is_a_over_n(self, ax, n):
    """The spacing is exactly ax / n, not ax / (n - 1)."""
    params = jax_ls.init_params(ax, ax, n, n, 40.0)
    spacing = float(params.X[0, 1] - params.X[0, 0])
    assert spacing == pytest.approx(ax / n, rel=1e-12)
    assert float(params.hx) == pytest.approx(ax / n, rel=1e-12)

  def test_stored_spacing_matches_the_grid(self, params_small):
    """`hx` must be the spacing of `X`, not an independent value."""
    assert float(params_small.hx) == pytest.approx(
      float(params_small.X[0, 1] - params_small.X[0, 0]), rel=1e-12
    )
    assert float(params_small.hy) == pytest.approx(
      float(params_small.Y[1, 0] - params_small.Y[0, 0]), rel=1e-12
    )

  def test_half_open_interval(self, params_small):
    """The grid covers [-a/2, a/2), so the last point is a/2 - h."""
    hx = float(params_small.hx)
    assert float(params_small.X.min()) == pytest.approx(-0.5, abs=1e-12)
    assert float(params_small.X.max()) == pytest.approx(0.5 - hx, abs=1e-12)

  def test_uniform_spacing(self, params_small):
    """Every step must be identical."""
    dx = jnp.diff(params_small.X[0, :])
    dy = jnp.diff(params_small.Y[:, 0])
    assert float(jnp.max(jnp.abs(dx - dx[0]))) < 1e-14
    assert float(jnp.max(jnp.abs(dy - dy[0]))) < 1e-14

  def test_cell_area(self, params_small):
    """`cell_area` is the quadrature weight of one cell."""
    assert float(params_small.cell_area) == pytest.approx(
      float(params_small.hx * params_small.hy), rel=1e-12
    )


class TestQuadratureWeightConsistency:
  """The stored spacing must be the one the FFT convolution implies."""

  @pytest.mark.parametrize("n", [2**6, 2**7])
  def test_weight_agrees_with_direct_quadrature(
    self, n, direct_volume_potential, nearest_grid_index
  ):
    """G*f by FFT must reproduce direct quadrature scaled by hx * hy.

    Away from the singularity the FFT convolution *is* the quadrature sum, so
    the agreement is at round-off; a wrong quadrature weight shows up as a
    relative error of order 2 / (n - 1).
    """
    params = jax_ls.init_params(1.0, 1.0, n, n, 16.0)

    f = jnp.exp(-200.0 * (jnp.square(params.X) + jnp.square(params.Y)))
    f_vect = jnp.reshape(f, (-1,)) + 0j
    u_fft = jax_ls.apply_green_function(params, f_vect)

    ix, iy = nearest_grid_index(params, 0.3, 0.2)
    got = complex(u_fft.reshape((n, n))[iy, ix])
    reference = direct_volume_potential(params, f_vect, ix, iy)

    err = abs(got - reference) / abs(reference)
    assert err < 1e-9, f"FFT vs quadrature error {err:.2e} exceeds 1e-9"

  def test_stored_weight_beats_the_endpoint_convention(
    self, direct_volume_potential, nearest_grid_index
  ):
    """h = a/n must be a far better quadrature weight than h = a/(n-1)."""
    n = 2**7
    params = jax_ls.init_params(1.0, 1.0, n, n, 16.0)

    f = jnp.exp(-200.0 * (jnp.square(params.X) + jnp.square(params.Y)))
    f_vect = jnp.reshape(f, (-1,)) + 0j
    u_fft = jax_ls.apply_green_function(params, f_vect)

    ix, iy = nearest_grid_index(params, 0.3, 0.2)
    got = complex(u_fft.reshape((n, n))[iy, ix])
    reference = direct_volume_potential(params, f_vect, ix, iy)

    # Undo the correct weight and reapply the endpoint one.
    rescaled = reference * ((1.0 / (n - 1)) ** 2 / float(params.cell_area))

    err_new = abs(got - reference) / abs(got)
    err_old = abs(got - rescaled) / abs(got)
    assert err_new < 1e-9 < err_old, (
      f"h = a/n gives error {err_new:.3e}, h = a/(n-1) gives {err_old:.3e}"
    )


class TestRectangularDomain:
  """`ax != ay` and `n != m` must work, not just be accepted."""

  def test_grid_shapes(self, params_rect, rect_config):
    """X and Y have shape (m, n); the padded kernel is (pad*m, pad*n)."""
    n, m = rect_config["n"], rect_config["m"]

    assert params_rect.X.shape == (m, n)
    assert params_rect.Y.shape == (m, n)
    assert params_rect.GFFT.shape == (
      jax_ls.DEFAULT_PAD * m,
      jax_ls.DEFAULT_PAD * n,
    )

  def test_axes_are_independent(self, params_rect, rect_config):
    """The y axis uses m points over ay, not n points over ay."""
    ax, ay = rect_config["ax"], rect_config["ay"]
    n, m = rect_config["n"], rect_config["m"]

    assert float(params_rect.hx) == pytest.approx(ax / n, rel=1e-12)
    assert float(params_rect.hy) == pytest.approx(ay / m, rel=1e-12)
    assert float(params_rect.X.min()) == pytest.approx(-ax / 2, abs=1e-12)
    assert float(params_rect.Y.min()) == pytest.approx(-ay / 2, abs=1e-12)
    assert float(params_rect.Y.max()) == pytest.approx(
      ay / 2 - ay / m, abs=1e-12
    )

  def test_green_function_preserves_shape(self, params_rect):
    """The convolution must round-trip a rectangular grid."""
    u = jnp.ones(params_rect.n_dof, dtype=complex)
    assert jax_ls.apply_green_function(params_rect, u).shape == u.shape

  def test_green_adjoint_identity(self, params_rect, rng_key):
    """<Gu, v> = <u, G^H v> on a rectangular grid."""
    key1, key2 = jax.random.split(rng_key)
    size = params_rect.n_dof
    u = jax.random.normal(key1, (size,)) + 1j * jax.random.normal(key1, (size,))
    v = jax.random.normal(key2, (size,)) + 1j * jax.random.normal(key2, (size,))

    lhs = jnp.sum(jnp.conj(v) * jax_ls.apply_green_function(params_rect, u))
    rhs = jnp.sum(
      jnp.conj(jax_ls.apply_conj_green_function(params_rect, v)) * u
    )
    err = float(jnp.abs(lhs - rhs) / jnp.abs(lhs))
    assert err < 1e-10, f"rectangular adjoint error {err:.2e}"

  def test_solver_residual(self, params_rect, perturbation_fn):
    """GMRES must actually solve the system on a rectangular grid."""
    omega = float(params_rect.omega)
    nu = perturbation_fn(params_rect.X, params_rect.Y)
    nu_vect = jnp.reshape(nu, (-1,))

    u_i = jnp.exp(1j * omega * params_rect.Y)
    f = jnp.reshape(-(omega**2) * nu * u_i, (-1,))

    sigma = jax_ls.ls_solver(params_rect, nu_vect, f)
    residual = jax_ls.apply_lipp_schwin(params_rect, nu_vect, sigma) - f
    err = float(jnp.linalg.norm(residual) / jnp.linalg.norm(f))
    assert err < 1e-4, f"rectangular solver residual {err:.2e}"

  def test_near_field_shapes(
    self, params_nf_rect, rect_config, perturbation_fn
  ):
    """The near-field map must work on a rectangular grid."""
    m = rect_config["m"]

    ls = params_nf_rect.ls_params
    nu_vect = jnp.reshape(perturbation_fn(ls.X, ls.Y), (-1,))
    near_field = jax_ls.near_field_map(params_nf_rect, nu_vect)

    assert params_nf_rect.G_sample.shape == (ls.n_dof, m)
    assert near_field.shape == (m, m)
    assert bool(jnp.all(jnp.isfinite(jnp.abs(near_field))))

  def test_matches_a_square_embedding(self):
    """A rectangular domain must agree with the same source in a square box.

    This is the sharp test of the per-dimension padded periods: taking them
    from `max(ax, ay)` in both directions evaluates the kernel at the wrong
    wavenumbers along the short axis.  Both grids here share their spacing, so
    the rectangular points are a subset of the square ones and the two volume
    potentials must agree to round-off.
    """
    omega = 16.0
    rect = jax_ls.init_params(1.0, 0.5, 2**7, 2**6, omega)
    square = jax_ls.init_params(1.0, 1.0, 2**7, 2**7, omega)
    assert float(rect.hy) == pytest.approx(float(square.hy), rel=1e-12)

    def potential(params):
      f = jnp.exp(-400.0 * (jnp.square(params.X) + jnp.square(params.Y)))
      u = jax_ls.apply_green_function(params, jnp.reshape(f, (-1,)) + 0j)
      return u.reshape(params.X.shape)

    u_rect = potential(rect)
    # The rectangular y-grid is rows [offset, offset + m) of the square grid.
    offset = (square.X.shape[0] - rect.X.shape[0]) // 2
    u_square = potential(square)[offset : offset + rect.X.shape[0], :]

    err = float(jnp.linalg.norm(u_rect - u_square) / jnp.linalg.norm(u_square))
    assert err < 1e-9, f"rectangular vs square embedding error {err:.2e}"

  def test_volume_potential_with_anisotropic_spacing(
    self, direct_volume_potential, nearest_grid_index
  ):
    """The FFT convolution must be correct when hx != hy."""
    ax, ay, n, m, omega = 1.0, 0.5, 2**7, 2**5, 16.0
    params = jax_ls.init_params(ax, ay, n, m, omega)
    assert float(params.hx) != float(params.hy)

    f = jnp.exp(-400.0 * (jnp.square(params.X) + jnp.square(params.Y)))
    f_vect = jnp.reshape(f, (-1,)) + 0j
    u_fft = jax_ls.apply_green_function(params, f_vect)

    ix, iy = nearest_grid_index(params, 0.3, 0.15)
    got = complex(u_fft.reshape((m, n))[iy, ix])
    reference = direct_volume_potential(params, f_vect, ix, iy)

    err = abs(got - reference) / abs(reference)
    assert err < 1e-8, f"anisotropic volume potential error {err:.2e}"


class TestValidation:
  """`init_params` must reject configurations it cannot represent."""

  def test_rejects_tiny_grid(self):
    with pytest.raises(ValueError, match="at least 2 points"):
      jax_ls.init_params(1.0, 1.0, 1, 8, 10.0)

  def test_rejects_small_padding(self):
    with pytest.raises(ValueError, match="padding factor"):
      jax_ls.init_params(1.0, 1.0, 8, 8, 10.0, pad=1)

  def test_rejects_truncation_below_the_diameter(self):
    with pytest.raises(ValueError, match="smaller than the diameter"):
      jax_ls.init_params(1.0, 1.0, 8, 8, 10.0, truncation_radius=1.0)

  def test_rejects_aliasing_configuration(self):
    """A very anisotropic domain needs more padding than the default."""
    with pytest.raises(ValueError, match="padding 4 is too small"):
      jax_ls.init_params(4.0, 1.0, 32, 8, 10.0)

  def test_more_padding_rescues_anisotropy(self):
    """...and raising `pad` accepts it."""
    params = jax_ls.init_params(4.0, 1.0, 32, 8, 10.0, pad=8)
    assert np.isfinite(np.asarray(params.GFFT)).all()
