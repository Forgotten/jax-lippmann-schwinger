"""Unit tests for `SamplingManifold`, `IncidentField` and asymmetric geometries.

The point of the source/receiver split is that the two may now differ.  These
tests cover the manifolds themselves and, more importantly, the behaviour that
was *unreachable* before: a near-field map whose sources and receivers are
different sets.

Deliberately not covered here, because it already is elsewhere:

* the symmetric near-field map and its derivatives -- `test_near_field.py`,
  `test_derivatives.py`, `test_adjoint.py`, which all still pass unchanged and
  are what pins the split as backwards compatible;
* the sign and scale of the sampling kernel -- `test_near_field.py`.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import jax_ls
from jax_ls.domain import Domain
from jax_ls.sampling import PlaneWaves, PointSources, SamplingManifold

jax.config.update("jax_enable_x64", True)


class TestSamplingManifold:
  """Geometry only: where the points are."""

  def test_circle_matches_the_original_parametrisation(self):
    """Starts at pi and wraps once; pins the convention the old code used."""
    n_theta, radius = 8, 1.5
    d_theta = 2 * np.pi / n_theta
    theta = np.linspace(np.pi, 3 * np.pi - d_theta, n_theta)
    expected = radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)

    points = np.asarray(SamplingManifold.circle(radius, n_theta).points)
    assert np.max(np.abs(points - expected)) < 1e-12

  def test_circle_points_lie_on_the_circle(self):
    manifold = SamplingManifold.circle(2.0, 16, center=(0.5, -0.25))
    radii = jnp.linalg.norm(manifold.points - jnp.array([[0.5, -0.25]]), axis=1)
    assert float(jnp.max(jnp.abs(radii - 2.0))) < 1e-12

  def test_arc_spans_the_requested_aperture(self):
    """A limited aperture: endpoints inclusive, unlike the full circle."""
    arc = SamplingManifold.arc(1.0, 5, 0.0, jnp.pi / 2)
    angles = jnp.arctan2(arc.points[:, 1], arc.points[:, 0])

    assert len(arc) == 5
    assert float(angles[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(angles[-1]) == pytest.approx(np.pi / 2, abs=1e-12)

  def test_line_interpolates_between_the_endpoints(self):
    line = SamplingManifold.line((-1.0, 0.5), (1.0, 0.5), 5)

    assert float(line.points[0, 0]) == pytest.approx(-1.0)
    assert float(line.points[-1, 0]) == pytest.approx(1.0)
    assert bool(jnp.all(line.points[:, 1] == 0.5))

  def test_from_points_rejects_the_wrong_shape(self):
    with pytest.raises(ValueError, match=r"\(n_points, 2\)"):
      SamplingManifold.from_points(jnp.zeros((4, 3)))

  def test_distances_are_symmetric_in_the_two_arguments(self):
    """`distances_to` must be the plain Euclidean distance, not squared."""
    domain = Domain(1.0, 1.0, 4, 4)
    manifold = SamplingManifold.from_points(jnp.array([[2.0, 0.0]]))
    expected = jnp.hypot(domain.X.reshape(-1) - 2.0, domain.Y.reshape(-1))

    assert bool(
      jnp.allclose(manifold.distances_to(domain)[:, 0], expected, rtol=1e-14)
    )


class TestIncidentField:
  """What illuminates the medium."""

  def test_point_sources_reproduce_the_sampling_kernel(self):
    """A point source radiates the same kernel that samples it -- reciprocity."""
    domain = Domain(1.0, 1.0, 8, 8)
    manifold = SamplingManifold.circle(1.5, 6)

    assert bool(
      jnp.all(
        PointSources(manifold).evaluate(domain, 10.0)
        == manifold.sampling_kernel(domain, 10.0)
      )
    )

  def test_plane_waves_have_unit_modulus(self):
    """A plane wave does not decay; only its phase varies over the grid."""
    domain = Domain(1.0, 1.0, 8, 8)
    waves = PlaneWaves.from_angles(
      jnp.linspace(0, 2 * jnp.pi, 4, endpoint=False)
    )
    field = waves.evaluate(domain, 12.0)

    assert field.shape == (domain.n_dof, 4)
    assert bool(jnp.allclose(jnp.abs(field), 1.0, rtol=1e-12))

  def test_plane_wave_phase_advances_along_its_direction(self):
    """Pins the sign of the phase: exp(i omega d.x), up to the -1 convention."""
    domain = Domain(1.0, 1.0, 8, 8)
    omega = 12.0
    waves = PlaneWaves(directions=jnp.array([[1.0, 0.0]]))
    field = waves.evaluate(domain, omega)[:, 0]
    expected = -jnp.exp(1j * omega * domain.X.reshape(-1))

    assert bool(jnp.allclose(field, expected, rtol=1e-12))

  def test_n_sources_agrees_with_the_evaluated_field(self):
    domain = Domain(1.0, 1.0, 8, 8)
    for incident in (
      PointSources(SamplingManifold.circle(1.5, 7)),
      PlaneWaves.from_angles(jnp.linspace(0, jnp.pi, 5)),
    ):
      assert incident.evaluate(domain, 10.0).shape[1] == incident.n_sources


class TestAsymmetricGeometry:
  """Sources and receivers as genuinely different sets.

  None of this was expressible before: `G_sample` served both roles, so the
  shapes were forced equal and the adjoint silently relied on the near-field
  matrix being symmetric.
  """

  @pytest.fixture(scope="class")
  def params_split(self):
    """16 sources, 32 receivers -- deliberately not square."""
    return jax_ls.init_params_near_field(
      1.0, 1.0, 48, 48, 1.5, 16, 30.0, n_receivers=32
    )

  @pytest.fixture(scope="class")
  def contrast(self, params_split):
    ls = params_split.ls_params
    return ls.domain.sample(
      lambda X, Y: (
        0.6 * jnp.exp(-200 * (jnp.square(X + 0.05) + jnp.square(Y - 0.08)))
      )
    )

  def test_near_field_is_rectangular(self, params_split, contrast):
    """The data matrix is (n_sources, n_receivers), not square."""
    assert params_split.data_shape == (16, 32)
    assert jax_ls.near_field_map(params_split, contrast).shape == (16, 32)

  def test_reciprocity_is_reported_as_absent(self, params_split):
    """`is_reciprocal` must not claim symmetry that no longer holds."""
    assert not params_split.is_reciprocal

    symmetric = jax_ls.init_params_near_field(1.0, 1.0, 48, 48, 1.5, 16, 30.0)
    assert symmetric.is_reciprocal

  def test_n_theta_refuses_to_guess(self, params_split):
    """The ambiguous accessor must raise rather than pick one."""
    with pytest.raises(ValueError, match="ambiguous"):
      _ = params_split.n_theta

  def test_adjoint_identity_without_the_reciprocity_shortcut(
    self, params_split, contrast, rng_key
  ):
    """<w, J t> = <J^H w, t> when the symmetry argument is unavailable.

    The adjoint contracts the cotangent along its receiver index.  With square
    data either pairing works, because the range of J is symmetric; here only
    the correct one does, so this is the test that actually pins the transpose.
    """
    key_t, key_r, key_i = jax.random.split(rng_key, 3)
    n_dof = params_split.ls_params.n_dof
    n_data = params_split.n_sources * params_split.n_receivers

    tangent = jax.random.normal(key_t, (n_dof,))
    cotangent = jax.random.normal(key_r, (n_data,)) + 1j * jax.random.normal(
      key_i, (n_data,)
    )

    lhs = jnp.vdot(
      cotangent, jax_ls.born_linearization(params_split, contrast, tangent)
    )
    rhs = jnp.vdot(
      jax_ls.born_adjoint(params_split, contrast, cotangent), tangent
    )

    err = float(jnp.abs(lhs - rhs) / jnp.abs(lhs))
    assert err < 1e-8, f"adjoint identity broken on a split geometry: {err:.2e}"

  def test_gradient_matches_finite_differences(self, params_split, contrast):
    """The misfit gradient must survive the rectangular data shape."""
    data = jax_ls.near_field_map(params_split, contrast)
    loss = lambda v: jax_ls.near_field_l2_loss(params_split, data, v)  # noqa: E731

    direction = params_split.ls_params.domain.sample(
      lambda X, Y: jnp.exp(-400 * (jnp.square(X - 0.1) + jnp.square(Y - 0.1)))
    )
    start = 0.5 * contrast
    step = 1e-4

    directional = float(jnp.sum(jax.grad(loss)(start) * direction))
    difference = float(
      (loss(start + step * direction) - loss(start - step * direction))
      / (2 * step)
    )

    err = abs(directional - difference) / abs(difference)
    assert err < 1e-6, f"gradient vs finite differences {err:.2e}"

  def test_plane_waves_onto_a_receiver_arc(self, rng_key):
    """A transmission-style experiment: no shared geometry at all."""
    ls_params = jax_ls.init_params(1.0, 1.0, 48, 48, 30.0)
    params = jax_ls.make_near_field_params(
      ls_params,
      PlaneWaves.from_angles(jnp.linspace(0, 2 * jnp.pi, 12, endpoint=False)),
      SamplingManifold.arc(1.5, 20, 0.2, 2.4),
    )
    contrast = ls_params.domain.sample(
      lambda X, Y: 0.4 * jnp.exp(-200 * (jnp.square(X) + jnp.square(Y)))
    )

    assert params.data_shape == (12, 20)
    assert jax_ls.near_field_map(params, contrast).shape == (12, 20)

    key_t, key_r, key_i = jax.random.split(rng_key, 3)
    tangent = jax.random.normal(key_t, (ls_params.n_dof,))
    cotangent = jax.random.normal(key_r, (240,)) + 1j * jax.random.normal(
      key_i, (240,)
    )

    lhs = jnp.vdot(
      cotangent, jax_ls.born_linearization(params, contrast, tangent)
    )
    rhs = jnp.vdot(jax_ls.born_adjoint(params, contrast, cotangent), tangent)

    assert float(jnp.abs(lhs - rhs) / jnp.abs(lhs)) < 1e-8
