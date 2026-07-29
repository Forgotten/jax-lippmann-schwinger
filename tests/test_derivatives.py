"""Tests pinning the three differentiation entry points of the near-field map.

`near_field_map_forward_mode` carries a `custom_jvp` (the Born approximation),
`near_field_map_reverse_mode` carries a `custom_vjp` implementing the *transpose*
J^T (JAX's reverse-mode convention), and `born_adjoint` applies the *Hermitian*
adjoint J^H.  J^T and J^H differ by conjugation, and using one where the other is
expected produces a wrong answer with no error -- so each identity is asserted
here, including the fact that the two are genuinely different.
"""

from functools import partial

import pytest

import jax
import jax.numpy as jnp
from jax import jit, random

import jax_ls

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def setup(params_nf_small, perturbation_fn, delta_perturbation_fn):
  """Contrast, perturbation and jitted maps shared by the tests below."""
  params = params_nf_small
  ls = params.ls_params
  nu_vect = jnp.reshape(perturbation_fn(ls.X, ls.Y), (-1,))
  delta_nu_vect = jnp.reshape(delta_perturbation_fn(ls.X, ls.Y), (-1,))
  return {
    "params": params,
    "nu": nu_vect,
    "delta_nu": delta_nu_vect,
    "forward": jit(partial(jax_ls.near_field_map_forward_mode, params)),
    "reverse": jit(partial(jax_ls.near_field_map_reverse_mode, params)),
  }


class TestForwardMode:
  """`near_field_map_forward_mode` and its Born rule."""

  def test_agrees_with_the_matrix_map(self, setup):
    """The flattened map must equal `near_field_map` reshaped."""
    matrix = jax_ls.near_field_map(setup["params"], setup["nu"])
    flat = setup["forward"](setup["nu"])
    err = float(
      jnp.linalg.norm(matrix.reshape((-1,)) - flat) / jnp.linalg.norm(flat)
    )
    assert err < 1e-12, f"flat vs matrix map error {err:.2e}"

  def test_jvp_vs_finite_differences(self, setup):
    """The custom JVP must match a centred finite difference."""
    _, jacobian = jax.linearize(setup["forward"], setup["nu"])
    tangent = jacobian(setup["delta_nu"])

    delta_t = 0.01
    plus = setup["forward"](setup["nu"] + delta_t * setup["delta_nu"])
    minus = setup["forward"](setup["nu"] - delta_t * setup["delta_nu"])
    fd = (plus - minus) / (2 * delta_t)

    err = float(jnp.linalg.norm(tangent - fd) / jnp.linalg.norm(tangent))
    assert err < 1e-3, f"JVP vs FD error {err:.2e}"

  def test_born_linearization_matches_the_jvp(self, setup):
    """`born_linearization` is the same operator, applied explicitly."""
    _, jacobian = jax.linearize(setup["forward"], setup["nu"])
    expected = jacobian(setup["delta_nu"])
    got = jax_ls.born_linearization(
      setup["params"], setup["nu"], setup["delta_nu"]
    )
    err = float(jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected))
    assert err < 1e-12, f"born_linearization vs JVP error {err:.2e}"

  def test_linearization_is_reciprocal(self, setup):
    """J t is a symmetric matrix for every t.

    The linearised map is ``dF_{s,r} = -omega^2 int G_nu(x_r, y) t(y)
    G_nu(y, x_s) dy``, symmetric in (s, r) by reciprocity of the Green's
    function of the medium.  This is what licenses the index pairing inside
    `_adjoint_apply`: J annihilates the antisymmetric part of any cotangent.

    The residual asymmetry is set by the GMRES tolerance (1e-5 by default), not
    by the discretisation, hence the tolerance below.
    """
    n_theta = setup["params"].n_theta
    tangent = jax_ls.born_linearization(
      setup["params"], setup["nu"], setup["delta_nu"]
    ).reshape((n_theta, n_theta))
    err = float(jnp.linalg.norm(tangent - tangent.T) / jnp.linalg.norm(tangent))
    assert err < 1e-4, f"linearised map is not reciprocal, asymmetry {err:.2e}"


class TestHermitianAdjoint:
  """`born_adjoint` applies J^H."""

  def test_adjoint_identity(self, setup, rng_key):
    """<w, J t> = <J^H w, t> for the Hermitian inner product."""
    params, nu = setup["params"], setup["nu"]
    _, jacobian = jax.linearize(setup["forward"], nu)

    key1, key2 = random.split(rng_key)
    tangent = random.normal(key1, (nu.shape[0],))
    cotangent = random.normal(key2, (params.n_theta**2,)) + 1j * random.normal(
      key1, (params.n_theta**2,)
    )

    lhs = jnp.sum(jnp.conj(cotangent) * jacobian(tangent))
    rhs = jnp.sum(
      jnp.conj(jax_ls.born_adjoint(params, nu, cotangent)) * tangent
    )
    err = float(jnp.abs(lhs - rhs) / jnp.abs(lhs))
    assert err < 1e-6, f"Hermitian adjoint identity error {err:.2e}"

  def test_accepts_matrix_and_flat_cotangents(self, setup, rng_key):
    """The cotangent may be given flat or as a matrix."""
    params, nu = setup["params"], setup["nu"]
    n_theta = params.n_theta
    flat = random.normal(rng_key, (n_theta**2,)) + 0j

    from_flat = jax_ls.born_adjoint(params, nu, flat)
    from_matrix = jax_ls.born_adjoint(params, nu, flat.reshape((n_theta, -1)))
    assert float(jnp.linalg.norm(from_flat - from_matrix)) < 1e-12


class TestReverseMode:
  """`near_field_map_reverse_mode` applies J^T, which is what jax.grad needs."""

  def test_transpose_identity(self, setup, rng_key):
    """sum(w * J t) = sum((J^T w) * t), without conjugation."""
    params, nu = setup["params"], setup["nu"]
    _, jacobian = jax.linearize(setup["forward"], nu)
    _, vjp_fn = jax.vjp(setup["reverse"], nu)

    key1, key2 = random.split(rng_key)
    tangent = random.normal(key1, (nu.shape[0],))
    cotangent = random.normal(key2, (params.n_theta**2,)) + 1j * random.normal(
      key1, (params.n_theta**2,)
    )

    lhs = jnp.sum(cotangent * jacobian(tangent))
    rhs = jnp.sum(vjp_fn(cotangent)[0] * tangent)
    err = float(jnp.abs(lhs - rhs) / jnp.abs(lhs))
    assert err < 1e-6, f"transpose identity error {err:.2e}"

  def test_grad_of_a_real_loss_vs_finite_differences(self, setup):
    """`jax.grad` through the reverse-mode map must match FD.

    This is the identity the Hermitian adjoint does *not* satisfy.
    """
    nu, delta_nu = setup["nu"], setup["delta_nu"]
    data = setup["forward"](nu + 0.5 * delta_nu)

    @jit
    def loss(x):
      residual = setup["reverse"](x) - data
      return 0.5 * jnp.sum(jnp.abs(residual) ** 2)

    gradient = jax.grad(loss)(nu)
    directional = float(jnp.sum(jnp.real(gradient) * delta_nu))

    delta_t = 1e-3
    fd = float(
      (loss(nu + delta_t * delta_nu) - loss(nu - delta_t * delta_nu))
      / (2 * delta_t)
    )

    err = abs(directional - fd) / abs(fd)
    assert err < 1e-3, f"grad vs FD error {err:.2e}"

  def test_transpose_and_hermitian_are_different(self, setup, rng_key):
    """J^T and J^H must not be confused; they differ by conjugation.

    Regression guard: an earlier version exposed J^H through a `custom_vjp`,
    which made `jax.grad` silently wrong by ~90%.
    """
    params, nu = setup["params"], setup["nu"]
    _, vjp_fn = jax.vjp(setup["reverse"], nu)

    cotangent = random.normal(
      rng_key, (params.n_theta**2,)
    ) + 1j * random.normal(rng_key, (params.n_theta**2,))

    transpose = vjp_fn(cotangent)[0]
    hermitian = jax_ls.born_adjoint(params, nu, cotangent)

    # They agree only after the conjugation that relates the two conventions.
    related = jnp.conj(jax_ls.born_adjoint(params, nu, jnp.conj(cotangent)))
    assert (
      float(jnp.linalg.norm(transpose - related) / jnp.linalg.norm(transpose))
      < 1e-10
    ), "J^T should equal conj(J^H conj(.))"

    assert (
      float(jnp.linalg.norm(transpose - hermitian) / jnp.linalg.norm(transpose))
      > 1e-2
    ), "J^T and J^H should not coincide for a complex cotangent"


class TestLossGradient:
  """`near_field_l2_loss` and its fused value-and-gradient."""

  def test_gradient_is_real(self, setup):
    """The contrast is real, so the gradient must be too."""
    params, nu, delta_nu = setup["params"], setup["nu"], setup["delta_nu"]
    data = jax_ls.near_field_map(params, nu + 0.5 * delta_nu)
    gradient = jax.grad(partial(jax_ls.near_field_l2_loss, params, data))(nu)
    assert (
      not jnp.iscomplexobj(gradient)
      or float(jnp.linalg.norm(jnp.imag(gradient))) == 0.0
    )

  def test_value_and_grad_matches_autodiff(self, setup):
    """The hand-written value-and-gradient must match `jax.value_and_grad`."""
    params, nu, delta_nu = setup["params"], setup["nu"], setup["delta_nu"]
    data = jax_ls.near_field_map(params, nu + 0.5 * delta_nu)

    loss_fn = partial(jax_ls.near_field_l2_loss, params, data)
    expected_loss, expected_grad = jax.value_and_grad(loss_fn)(nu)
    loss, gradient = jax_ls.near_field_l2_loss_and_grad(params, data, nu)

    assert (
      float(jnp.abs(loss - expected_loss))
      < 1e-12 * float(jnp.abs(expected_loss)) + 1e-30
    )
    err = float(
      jnp.linalg.norm(gradient - expected_grad) / jnp.linalg.norm(expected_grad)
    )
    assert err < 1e-12, f"fused gradient differs by {err:.2e}"


class TestDeprecatedAliases:
  """The old names must keep resolving, to the corrected implementations."""

  def test_aliases_point_at_the_new_functions(self):
    assert jax_ls.near_field_map_vect is jax_ls.near_field_map_forward_mode
    assert jax_ls.near_field_map_vect_jax is jax_ls.near_field_map_reverse_mode
    assert jax_ls.near_field_map_vect_vjp is jax_ls.near_field_map_reverse_mode
    assert jax_ls.ls_solver_batched is jax_ls.ls_solver
    assert jax_ls.ls_solver_batched_sigma is jax_ls.ls_solver
    assert jax_ls.ls_solver_batched_adj is jax_ls.ls_solver_adj
    assert jax_ls.ls_solver_u is jax_ls.ls_solver_field


class TestLinearization:
  """`linearize` caches the forward solve that J and J^H both need.

  The identities themselves are asserted above; what is new here is that the
  cached object reproduces them *exactly*, so the optimisation is invisible.
  """

  def test_jvp_reproduces_born_linearization(self, setup):
    lin = jax_ls.linearize(setup["params"], setup["nu"])
    cached = lin.jvp(setup["delta_nu"])
    direct = jax_ls.born_linearization(
      setup["params"], setup["nu"], setup["delta_nu"]
    )

    err = float(jnp.linalg.norm(cached - direct) / jnp.linalg.norm(direct))
    assert err < 1e-12, f"cached J differs from born_linearization: {err:.2e}"

  def test_vjp_reproduces_born_adjoint(self, setup, rng_key):
    params = setup["params"]
    cotangent = (
      random.normal(rng_key, (params.n_sources * params.n_receivers,)) + 0j
    )

    lin = jax_ls.linearize(params, setup["nu"])
    cached = lin.vjp(cotangent)
    direct = jax_ls.born_adjoint(params, setup["nu"], cotangent)

    err = float(jnp.linalg.norm(cached - direct) / jnp.linalg.norm(direct))
    assert err < 1e-12, f"cached J^H differs from born_adjoint: {err:.2e}"

  def test_data_is_the_forward_map(self, setup):
    """The forward solve is done anyway, so F[nu] comes along free."""
    lin = jax_ls.linearize(setup["params"], setup["nu"])
    expected = jax_ls.near_field_map(setup["params"], setup["nu"]).reshape(
      (-1,)
    )

    err = float(
      jnp.linalg.norm(lin.data - expected) / jnp.linalg.norm(expected)
    )
    assert err < 1e-12

  def test_adjoint_identity_off_the_cached_object(self, setup, rng_key):
    """<w, J t> = <J^H w, t> using only the cached applications."""
    params = setup["params"]
    key_t, key_r, key_i = random.split(rng_key, 3)
    n_data = params.n_sources * params.n_receivers

    tangent = random.normal(key_t, (params.ls_params.n_dof,))
    cotangent = random.normal(key_r, (n_data,)) + 1j * random.normal(
      key_i, (n_data,)
    )

    lin = jax_ls.linearize(params, setup["nu"])
    lhs = jnp.vdot(cotangent, lin.jvp(tangent))
    rhs = jnp.vdot(lin.vjp(cotangent), tangent)

    err = float(jnp.abs(lhs - rhs) / jnp.abs(lhs))
    assert err < 1e-8, (
      f"adjoint identity broken on the cached object: {err:.2e}"
    )

  def test_works_on_an_asymmetric_geometry(self, rng_key):
    """Caching must not reintroduce the square-shape assumption."""
    params = jax_ls.init_params_near_field(
      1.0, 1.0, 32, 32, 1.5, 12, 25.0, n_receivers=20
    )
    contrast = params.ls_params.domain.sample(
      lambda X, Y: 0.5 * jnp.exp(-200 * (jnp.square(X) + jnp.square(Y)))
    )
    key_t, key_r = random.split(rng_key)

    lin = jax_ls.linearize(params, contrast)
    tangent = random.normal(key_t, (params.ls_params.n_dof,))
    cotangent = random.normal(key_r, (12 * 20,)) + 0j

    assert lin.jvp(tangent).shape == (12 * 20,)
    assert lin.vjp(cotangent).shape == (params.ls_params.n_dof,)

  def test_survives_a_jit_boundary(self, setup):
    """It is a pytree, so it can be built inside one jit and used in another."""
    build = jit(partial(jax_ls.linearize, setup["params"]))
    apply_jvp = jit(lambda lin, d: lin.jvp(d))

    lin = build(setup["nu"])
    assert apply_jvp(lin, setup["delta_nu"]).shape == lin.data.shape
