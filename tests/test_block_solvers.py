"""Unit tests for block GMRES and the block solve path of the near-field map.

Block GMRES solves ``A X = B`` for every right-hand side in one Krylov space.
It is an *option*, not the default, for reasons measured in
`jax_ls.solvers.BlockGmresConfig`; what these tests pin is that choosing it
changes performance and nothing else.

Deliberately not retested here, because it already is elsewhere: the per-source
identities themselves (`test_derivatives.py`, `test_sampling.py`).  The tests
below assert that the block path *reproduces* them.
"""

import pytest

import jax
import jax.numpy as jnp

import jax_ls
from jax_ls.solvers import BlockGmresConfig, block_solve, block_solve_with_info

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def dense_system():
  """A small non-Hermitian system with several right-hand sides."""
  n, p = 40, 4
  key = jax.random.PRNGKey(0)
  k1, k2, k3 = jax.random.split(key, 3)
  matrix = (
    jax.random.normal(k1, (n, n)) + 1j * jax.random.normal(k2, (n, n))
  ) / n**0.5 + 3.0 * jnp.eye(n)
  rhs = jax.random.normal(k3, (n, p)) + 0j
  return matrix, rhs


class TestBlockGmres:
  def test_config_is_all_static(self):
    """Settings are configuration, not data."""
    assert jax.tree.leaves(BlockGmresConfig()) == []

  def test_matches_a_dense_solve_once_the_space_is_full(self, dense_system):
    """With ``m * p >= n`` the Krylov space spans everything, so it is exact."""
    matrix, rhs = dense_system
    n, p = rhs.shape
    solution = block_solve(
      lambda X: matrix @ X, rhs, BlockGmresConfig(n_steps=n // p)
    )
    exact = jnp.linalg.solve(matrix, rhs)
    err = float(jnp.linalg.norm(solution - exact) / jnp.linalg.norm(exact))
    assert err < 1e-10, f"block GMRES vs dense solve: {err:.2e}"

  def test_residual_falls_as_the_block_space_grows(self, dense_system):
    """More block-Arnoldi steps must not make the answer worse."""
    matrix, rhs = dense_system
    residuals = []
    for steps in (2, 5, 8):
      solution = block_solve(
        lambda X: matrix @ X, rhs, BlockGmresConfig(n_steps=steps)
      )
      residuals.append(
        float(jnp.linalg.norm(matrix @ solution - rhs) / jnp.linalg.norm(rhs))
      )
    assert residuals[1] < residuals[0] and residuals[2] < residuals[1], (
      residuals
    )

  def test_restart_cycles_improve_on_one_cycle(self, dense_system):
    matrix, rhs = dense_system
    one = block_solve(
      lambda X: matrix @ X, rhs, BlockGmresConfig(n_steps=3, cycles=1)
    )
    three = block_solve(
      lambda X: matrix @ X, rhs, BlockGmresConfig(n_steps=3, cycles=3)
    )
    res = lambda X: float(  # noqa: E731
      jnp.linalg.norm(matrix @ X - rhs) / jnp.linalg.norm(rhs)
    )
    assert res(three) < res(one)

  def test_warm_start_is_accepted(self, dense_system):
    """An initial guess must be used, not ignored."""
    matrix, rhs = dense_system
    config = BlockGmresConfig(n_steps=3)
    cold = block_solve(lambda X: matrix @ X, rhs, config)
    warm = block_solve(lambda X: matrix @ X, rhs, config, x0=cold)
    res = lambda X: float(  # noqa: E731
      jnp.linalg.norm(matrix @ X - rhs) / jnp.linalg.norm(rhs)
    )
    assert res(warm) < res(cold)

  def test_info_reports_the_measured_residual(self, dense_system):
    matrix, rhs = dense_system
    solution, info = block_solve_with_info(
      lambda X: matrix @ X, rhs, BlockGmresConfig(n_steps=8)
    )
    expected = jnp.linalg.norm(matrix @ solution - rhs) / jnp.linalg.norm(rhs)
    assert float(info.relative_residual) == pytest.approx(
      float(expected), rel=1e-10
    )


class TestBlockPathInTheNearFieldMap:
  """Selecting the block solver must change speed only, not answers."""

  N = 32
  N_SRC = 10

  @pytest.fixture(scope="class")
  def pair(self):
    """The same problem with the per-source and the block solver."""
    omega = 2 * jnp.pi * (self.N // 8)
    common = (1.0, 1.0, self.N, self.N, 1.5, self.N_SRC, float(omega))
    per_source = jax_ls.init_params_near_field(*common)
    block = jax_ls.init_params_near_field(
      *common, block=BlockGmresConfig(n_steps=16)
    )
    contrast = per_source.ls_params.domain.sample(
      lambda X, Y: 0.5 * jnp.exp(-100 * (jnp.square(X) + jnp.square(Y)))
    )
    return per_source, block, contrast

  def test_block_is_opt_in(self, pair):
    per_source, block, _ = pair
    assert per_source.block is None
    assert block.block is not None

  def test_forward_map_agrees(self, pair):
    per_source, block, contrast = pair
    a = jax_ls.near_field_map(per_source, contrast)
    b = jax_ls.near_field_map(block, contrast)
    err = float(jnp.linalg.norm(b - a) / jnp.linalg.norm(a))
    assert err < 1e-9, f"block vs per-source forward map: {err:.2e}"

  def test_adjoint_identity_holds_on_the_block_path(self, pair, rng_key):
    """<w, J t> = <J^H w, t>, with both solves going through block GMRES."""
    _, block, contrast = pair
    key_t, key_r, key_i = jax.random.split(rng_key, 3)
    n_data = block.n_sources * block.n_receivers

    tangent = jax.random.normal(key_t, (block.ls_params.n_dof,))
    cotangent = jax.random.normal(key_r, (n_data,)) + 1j * jax.random.normal(
      key_i, (n_data,)
    )

    lhs = jnp.vdot(
      cotangent, jax_ls.born_linearization(block, contrast, tangent)
    )
    rhs = jnp.vdot(jax_ls.born_adjoint(block, contrast, cotangent), tangent)
    err = float(jnp.abs(lhs - rhs) / jnp.abs(lhs))
    assert err < 1e-8, f"adjoint identity on the block path: {err:.2e}"

  def test_gradient_matches_finite_differences(self, pair):
    per_source, block, contrast = pair
    data = jax_ls.near_field_map(per_source, contrast)
    domain = block.ls_params.domain
    direction = domain.sample(
      lambda X, Y: jnp.exp(-400 * (jnp.square(X - 0.1) + jnp.square(Y - 0.1)))
    )
    loss = lambda v: jax_ls.near_field_l2_loss(block, data, v)  # noqa: E731

    start, step = 0.5 * contrast, 1e-4
    directional = float(jnp.sum(jax.grad(loss)(start) * direction))
    difference = float(
      (loss(start + step * direction) - loss(start - step * direction))
      / (2 * step)
    )
    err = abs(directional - difference) / abs(difference)
    assert err < 1e-6, f"block-path gradient vs finite differences: {err:.2e}"
