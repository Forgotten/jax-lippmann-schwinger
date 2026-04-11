"""Tests for the Born approximation (linearisation) of the LS solver.

Compares the custom JVP (Born approximation) against centered
finite-difference approximations for both the wavefield solver
and the near-field map.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import jit, jvp, random
from functools import partial

import jax_ls


class TestBornApproximation:
  """Born approximation tests via JVP vs. finite-difference."""

  # Tolerances are now computed dynamically inside the tests based on delta_t to match the experiment scale.

  def test_born_approximation_ls_solver_u(self, perturbation_fn, delta_perturbation_fn):
    """JVP of ls_solver_u should match centered FD."""
    n = 2**7
    omega = 2 * jnp.pi * (n // 6)
    params = jax_ls.init_params(1.0, 1.0, n, n, omega)

    nu = perturbation_fn(params.X, params.Y) + 0j
    nu_vect = jnp.reshape(nu, (-1,))

    delta_nu = delta_perturbation_fn(params.X, params.Y) + 0j
    delta_nu_vect = jnp.reshape(delta_nu, (-1,))

    # incident field
    s = jnp.array([0.0, 1.0])
    s = s / jnp.linalg.norm(s)
    u_i = jnp.exp(1j * omega * (params.X * s[0] + params.Y * s[1]))
    rhs = -(omega**2) * nu * u_i
    f = jnp.reshape(rhs, (-1,))

    @jit
    def solver_u(nu_v):
      return jax_ls.ls_solver_u(params, nu_v, f)

    # centered finite difference
    delta_t = 0.01
    u_p = solver_u(nu_vect + delta_t * delta_nu_vect)
    u_m = solver_u(nu_vect - delta_t * delta_nu_vect)
    fd_approx = (u_p - u_m) / (2 * delta_t)

    # JVP (Born approximation)
    _, u_born = jvp(solver_u, (nu_vect,), (delta_nu_vect,))

    err = float(jnp.linalg.norm(fd_approx - u_born) / jnp.linalg.norm(u_born))
    # Dynamic tolerance based on finite difference step size delta_t
    # (FD error scales with O(delta_t^2) for centered differences, plus some baseline margin)
    tol = 0.005 + 10.0 * (delta_t**2)
    assert err < tol, (
      f"Born LS error {err:.2e} exceeds dynamic tolerance {tol:.2e} "
      f"(based on delta_t={delta_t})"
    )

  def test_born_approximation_near_field(
    self, params_nf_born, perturbation_fn, delta_perturbation_fn
  ):
    """JVP of near_field_map_vect should match centered FD."""
    params_nf = params_nf_born

    nu = perturbation_fn(params_nf.ls_params.X, params_nf.ls_params.Y) + 0j
    nu_vect = jnp.reshape(nu, (-1,))

    delta_nu = delta_perturbation_fn(params_nf.ls_params.X, params_nf.ls_params.Y) + 0j
    delta_nu_vect = jnp.reshape(delta_nu, (-1,))

    near_field = jit(partial(jax_ls.near_field_map_vect, params_nf))

    # JVP (Born)
    _, J = jax.linearize(near_field, nu_vect)
    near_field_born = J(delta_nu_vect)

    # centered finite difference
    delta_t = 0.01
    nf_p = near_field(nu_vect + delta_t * delta_nu_vect)
    nf_m = near_field(nu_vect - delta_t * delta_nu_vect)
    nf_fd = (nf_p - nf_m) / (2 * delta_t)

    err = float(
      jnp.linalg.norm(near_field_born - nf_fd) / jnp.linalg.norm(near_field_born)
    )
    # Dynamic near-field tolerance based on FD step size
    tol = 0.005 + 10.0 * (delta_t**2)
    assert err < tol, (
      f"Born near-field error {err:.2e} exceeds dynamic tolerance {tol:.2e} "
      f"(based on delta_t={delta_t})"
    )
