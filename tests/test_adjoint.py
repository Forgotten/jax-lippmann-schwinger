"""Adjoint consistency of the near-field operator at higher resolution.

`test_derivatives.py` pins the adjoint identities at 64x64; this file repeats the
Hermitian check at 128x128, where the discretisation error is smaller, with a
tolerance tied to the grid spacing.
"""

from functools import partial


import jax
import jax.numpy as jnp
from jax import jit, random

import jax_ls

jax.config.update("jax_enable_x64", True)


class TestAdjoint:
  """Adjoint test for `born_adjoint` against the forward-mode Jacobian."""

  def test_adjoint_consistency(
    self, params_nf_large, shifted_perturbation_fn, rng_key
  ):
    """<w, J t> should equal <J^H w, t> up to discretisation error."""
    params_nf = params_nf_large
    ls = params_nf.ls_params

    nu_vect = jnp.reshape(shifted_perturbation_fn(ls.X, ls.Y), (-1,))

    near_field = jit(partial(jax_ls.near_field_map_forward_mode, params_nf))
    _, jacobian = jax.linearize(near_field, nu_vect)

    key1, key2 = random.split(rng_key)
    tangent = random.normal(key1, shape=(nu_vect.shape[0],))
    cotangent = random.normal(
      key2, shape=(params_nf.n_theta**2,)
    ) + 1j * random.normal(key2, shape=(params_nf.n_theta**2,))

    lhs = jnp.sum(jnp.conj(cotangent) * jacobian(tangent))
    rhs = jnp.sum(
      jnp.conj(jax_ls.born_adjoint(params_nf, nu_vect, cotangent)) * tangent
    )
    err = float(jnp.abs(lhs - rhs) / jnp.abs(lhs))

    # GMRES baseline error plus an O(hx) discretisation mismatch.
    tol = 1e-4 + float(params_nf.hx) * 0.2
    assert err < tol, (
      f"Adjoint test error {err:.2e} exceeds dynamic tolerance {tol:.2e} "
      f"(based on hx={float(params_nf.hx):.4f})"
    )

  def test_reverse_mode_consistency(
    self, params_nf_large, shifted_perturbation_fn, rng_key
  ):
    """The reverse-mode rule must transpose the same Jacobian."""
    params_nf = params_nf_large
    ls = params_nf.ls_params

    nu_vect = jnp.reshape(shifted_perturbation_fn(ls.X, ls.Y), (-1,))

    forward = jit(partial(jax_ls.near_field_map_forward_mode, params_nf))
    reverse = jit(partial(jax_ls.near_field_map_reverse_mode, params_nf))

    _, jacobian = jax.linearize(forward, nu_vect)
    _, vjp_fn = jax.vjp(reverse, nu_vect)

    key1, key2 = random.split(rng_key)
    tangent = random.normal(key1, shape=(nu_vect.shape[0],))
    cotangent = random.normal(
      key2, shape=(params_nf.n_theta**2,)
    ) + 1j * random.normal(key2, shape=(params_nf.n_theta**2,))

    lhs = jnp.sum(cotangent * jacobian(tangent))
    rhs = jnp.sum(vjp_fn(cotangent)[0] * tangent)
    err = float(jnp.abs(lhs - rhs) / jnp.abs(lhs))

    tol = 1e-4 + float(params_nf.hx) * 0.2
    assert err < tol, (
      f"Transpose test error {err:.2e} exceeds dynamic tolerance {tol:.2e}"
    )
