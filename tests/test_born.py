"""Tests for the Born approximation (linearisation) of the LS solver.

Compares the custom JVP against centered finite differences for both the
wavefield solver and the near-field map.
"""

from functools import partial


import jax
import jax.numpy as jnp
from jax import jit, jvp

import jax_ls


class TestBornApproximation:
  """Born approximation tests via JVP vs. finite-difference."""

  def test_born_approximation_ls_solver_field(
    self, perturbation_fn, delta_perturbation_fn
  ):
    """JVP of ls_solver_field should match centered FD."""
    n = 2**7
    omega = 2 * jnp.pi * (n // 6)
    params = jax_ls.init_params(1.0, 1.0, n, n, omega)

    nu = perturbation_fn(params.X, params.Y) + 0j
    nu_vect = jnp.reshape(nu, (-1,))

    delta_nu = delta_perturbation_fn(params.X, params.Y) + 0j
    delta_nu_vect = jnp.reshape(delta_nu, (-1,))

    # Incident plane wave travelling in +y.
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
    # Centred differences converge like O(delta_t^2), plus a GMRES floor.
    tol = 0.005 + 10.0 * (delta_t**2)
    assert err < tol, (
      f"Born LS error {err:.2e} exceeds dynamic tolerance {tol:.2e} "
      f"(based on delta_t={delta_t})"
    )

  def test_born_approximation_near_field(
    self, params_nf_born, perturbation_fn, delta_perturbation_fn
  ):
    """JVP of near_field_map_forward_mode should match centered FD."""
    params_nf = params_nf_born
    ls = params_nf.ls_params

    nu_vect = jnp.reshape(perturbation_fn(ls.X, ls.Y) + 0j, (-1,))
    delta_nu_vect = jnp.reshape(delta_perturbation_fn(ls.X, ls.Y) + 0j, (-1,))

    near_field = jit(partial(jax_ls.near_field_map_forward_mode, params_nf))

    _, jacobian = jax.linearize(near_field, nu_vect)
    near_field_born = jacobian(delta_nu_vect)

    delta_t = 0.01
    nf_p = near_field(nu_vect + delta_t * delta_nu_vect)
    nf_m = near_field(nu_vect - delta_t * delta_nu_vect)
    nf_fd = (nf_p - nf_m) / (2 * delta_t)

    err = float(
      jnp.linalg.norm(near_field_born - nf_fd)
      / jnp.linalg.norm(near_field_born)
    )
    tol = 0.005 + 10.0 * (delta_t**2)
    assert err < tol, (
      f"Born near-field error {err:.2e} exceeds dynamic tolerance {tol:.2e} "
      f"(based on delta_t={delta_t})"
    )
