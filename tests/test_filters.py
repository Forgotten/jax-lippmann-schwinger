"""Unit tests for the band-limiting projector.

`band_limit` has to be an *orthogonal projector*, because `gauss_newton_step`
composes it into both halves of a least-squares operator pair and CG/CGLS are
only valid if those remain a genuine adjoint pair.  Idempotence and
self-adjointness are therefore correctness requirements, not niceties.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import jax_ls
from jax_ls.domain import Domain

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def domain():
  return Domain(1.0, 1.0, 32, 32)


@pytest.fixture(scope="module")
def field(domain):
  return jax.random.normal(jax.random.PRNGKey(0), (domain.n_dof,))


class TestProjector:
  OMEGA = 8 * np.pi

  def test_is_idempotent(self, domain, field):
    once = jax_ls.band_limit(domain, self.OMEGA, field)
    twice = jax_ls.band_limit(domain, self.OMEGA, once)

    err = float(jnp.linalg.norm(twice - once) / jnp.linalg.norm(once))
    assert err < 1e-13, f"not a projector: P^2 != P by {err:.2e}"

  def test_is_self_adjoint(self, domain, field):
    other = jax.random.normal(jax.random.PRNGKey(1), (domain.n_dof,))

    lhs = float(jnp.vdot(jax_ls.band_limit(domain, self.OMEGA, field), other))
    rhs = float(jnp.vdot(field, jax_ls.band_limit(domain, self.OMEGA, other)))

    assert lhs == pytest.approx(rhs, rel=1e-11)

  def test_keeps_a_real_field_real(self, domain, field):
    """The retained mode set is symmetric under xi -> -xi."""
    filtered = jax_ls.band_limit(domain, self.OMEGA, field)
    assert not jnp.iscomplexobj(filtered)


class TestWhatItKeeps:
  def test_passes_a_mode_inside_the_band_and_kills_one_outside(self, domain):
    """A pure cosine survives iff its wavenumber is below the limit."""
    omega = 8 * np.pi
    limit = 2 * omega  # the Born band |xi| <= 2 omega

    def cosine(wavenumber):
      return domain.sample(lambda X, Y: jnp.cos(wavenumber * X))

    inside = cosine(0.5 * limit)
    outside = cosine(2.0 * limit)

    kept = jax_ls.band_limit(domain, omega, inside)
    removed = jax_ls.band_limit(domain, omega, outside)

    assert (
      float(jnp.linalg.norm(kept - inside) / jnp.linalg.norm(inside)) < 1e-12
    )
    assert float(jnp.linalg.norm(removed) / jnp.linalg.norm(outside)) < 1e-12

  def test_dimension_matches_the_area_of_the_disc(self, domain):
    """The count of retained modes is ~ pi * (omega a / pi)^2."""
    omega = 8 * np.pi
    predicted = np.pi * jax_ls.band_index(domain, omega) ** 2
    actual = jax_ls.band_dimension(domain, omega)

    assert actual == pytest.approx(predicted, rel=0.05)
    assert actual < domain.n_dof  # the whole point: far fewer unknowns

  def test_band_grows_with_frequency(self, domain):
    dimensions = [
      jax_ls.band_dimension(domain, omega)
      for omega in (2 * np.pi, 4 * np.pi, 8 * np.pi)
    ]
    assert dimensions[0] < dimensions[1] < dimensions[2]

  def test_index_is_the_documented_formula(self, domain):
    omega = 8 * np.pi
    assert float(jax_ls.band_index(domain, omega)) == pytest.approx(
      2 * omega * domain.ax / (2 * np.pi)
    )
