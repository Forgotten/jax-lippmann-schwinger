"""Tests for the adjoint consistency of the near-field operator.

Verifies that <J δν, w> ≈ <δν, J* w> for random tangent/cotangent
vectors, using both the automatic (JAX linearize) and custom VJP
implementations.
"""

import jax_ls
from functools import partial
from jax import jit, random
import jax.numpy as jnp
import pytest
import jax

# Enable 64-bit support
jax.config.update("jax_enable_x64", True)


class TestAdjoint:
  """Adjoint-test for near_field_map_vect_vjp."""

  # The error tolerance is now computed dynamically based on the grid spacing

  def test_adjoint_consistency(
    self, params_nf_large, shifted_perturbation_fn, delta_perturbation_fn, rng_key
  ):
    """<J δν, w> should equal <δν, J* w> up to discretisation error."""
    params_nf = params_nf_large

    # sample perturbations on the grid
    nu = shifted_perturbation_fn(params_nf.ls_params.X, params_nf.ls_params.Y)
    nu_vect = jnp.reshape(nu, (-1,))

    delta_nu = delta_perturbation_fn(params_nf.ls_params.X, params_nf.ls_params.Y)
    delta_nu_vect = jnp.reshape(delta_nu, (-1,))

    # forward near-field (automatic JVP path)
    near_field = jit(partial(jax_ls.near_field_map_vect, params_nf))
    y, J = jax.linearize(near_field, nu_vect)

    # custom VJP path
    near_field_vjp = jit(partial(jax_ls.near_field_map_vect_vjp, params_nf))
    primals, J_star = jax.vjp(near_field_vjp, nu_vect)

    # random cotangent / tangent
    key1, key2, key3 = random.split(rng_key, 3)
    delta_u_nf = J(jax.random.normal(key1, shape=(y.shape[0],)))
    rand_cotgnt = jax.random.normal(
      key2, shape=(delta_u_nf.shape[0],)
    ) + 1j * jax.random.normal(key2, shape=(delta_u_nf.shape[0],))
    rand_tgnt = jax.random.normal(key3, shape=(delta_u_nf.shape[0],))

    # apply J (linearised) and J* (adjoint)
    delta_u_nf = J(rand_tgnt)
    adjoint = J_star(rand_cotgnt)

    # adjoint identity check
    lhs = jnp.sum(jnp.conj(rand_cotgnt) * delta_u_nf)
    rhs = jnp.sum(jnp.conj(adjoint[0]) * rand_tgnt)
    err = jnp.abs(lhs - rhs) / jnp.abs(lhs)

    # Compute dynamic tolerance: GMRES baseline error + O(hx) discretisation mismatch
    tol = 1e-4 + float(params_nf.hx) * 0.2
    assert float(err) < tol, (
      f"Adjoint test error {float(err):.2e} exceeds dynamic tolerance "
      f"{tol:.2e} (based on hx={float(params_nf.hx):.3f})"
    )
