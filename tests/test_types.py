"""Unit tests for the `Domain`, `GreenKernel` and `LippSchwinParams` types.

These cover the dataclasses *as objects*: their pytree structure, the contracts
between them, and the constructor behaviour that is not reachable through
`init_params`.

Deliberately **not** covered here, because it already is elsewhere:

* the grid convention itself (spacing `a/n`, half-open interval, `cell_area`)
  -- `test_grid.py::TestGridConventions`;
* rejection of a tiny grid, small padding, a truncation radius below the
  diameter, and aliasing configurations -- `test_grid.py::TestValidation`,
  which exercises the same validators through `init_params`;
* what the kernel computes -- `test_kernel.py` and `test_vico_greengard.py`.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import jax_ls
from jax_ls.domain import Domain
from jax_ls.kernels import DEFAULT_TRUNCATION_FACTOR, GreenKernel

jax.config.update("jax_enable_x64", True)


class TestDomain:
  """`Domain` is pure geometry and carries no data."""

  def test_has_no_array_leaves(self):
    """The invariant that lets one domain be shared by every frequency.

    If any field were a pytree node it would be duplicated per frequency and,
    worse, traced -- see `test_rejects_traced_use` below for why that breaks.
    """
    assert jax.tree.leaves(Domain(1.0, 0.5, 8, 4)) == []

  def test_survives_a_jit_boundary_as_a_static_argument(self):
    """Static fields stay concrete, so shapes may depend on them."""
    domain = Domain(1.0, 1.0, 8, 8)

    @jax.jit
    def dofs(d):
      # Would raise ConcretizationTypeError if `n` were a traced leaf.
      return jnp.arange(d.n).sum()

    assert int(dofs(domain)) == 28

  def test_tree_map_leaves_geometry_untouched(self):
    """Geometry must survive the `zeros_like` sweeps autodiff performs."""
    domain = Domain(1.0, 0.5, 8, 4)
    zeroed = jax.tree.map(lambda leaf: leaf * 0, domain)
    assert zeroed == domain

  def test_coordinates_match_meshgrid(self):
    """`X`/`Y` must equal `jnp.meshgrid(x, y)`.

    Pins the (m, n) row-major ordering that every downstream `reshape((-1,))`
    depends on; transposing it would silently mirror the medium.
    """
    domain = Domain(1.0, 0.5, 8, 4)
    X, Y = jnp.meshgrid(domain.x, domain.y)

    assert domain.X.shape == domain.Y.shape == domain.shape
    assert bool(jnp.all(domain.X == X))
    assert bool(jnp.all(domain.Y == Y))

  def test_grid_and_ravel_are_inverse(self):
    domain = Domain(1.0, 0.5, 8, 4)
    flat = jnp.arange(domain.n_dof, dtype=float)

    assert domain.grid(flat).shape == domain.shape
    assert bool(jnp.all(domain.ravel(domain.grid(flat)) == flat))

  def test_sample_evaluates_on_the_grid(self):
    """`sample` replaces the `fn(X, Y).reshape(-1)` idiom."""
    domain = Domain(1.0, 0.5, 8, 4)
    fn = lambda x, y: x + 2.0 * y  # noqa: E731

    assert bool(
      jnp.all(domain.sample(fn) == domain.ravel(fn(domain.X, domain.Y)))
    )

  def test_rejects_a_non_positive_size(self):
    """Not reachable through `init_params`' own checks."""
    with pytest.raises(ValueError, match="positive size"):
      Domain(0.0, 1.0, 8, 8)


class TestGreenKernel:
  """`GreenKernel` owns the frequency-dependent data and its domain."""

  def test_has_exactly_one_array_leaf(self):
    """`ghat` is the only data; the domain and the config ride along free."""
    kernel = GreenKernel.create(Domain(1.0, 1.0, 8, 8), 10.0)
    leaves = jax.tree.leaves(kernel)

    assert len(leaves) == 1
    assert leaves[0].shape == (kernel.pad * 8, kernel.pad * 8)

  def test_carries_its_domain_so_apply_needs_no_shape(self):
    """The composition contract: cropping is derived, not passed in."""
    domain = Domain(1.0, 0.5, 16, 8)
    kernel = GreenKernel.create(domain, 10.0)

    assert kernel.domain == domain
    assert kernel.apply(jnp.ones(domain.n_dof, dtype=complex)).shape == (
      domain.n_dof,
    )

  def test_free_functions_delegate_to_the_kernel(self, params_small):
    """`apply_green_function` must be the kernel method, not a second copy.

    Not bitwise equality: the free functions are `@jit`-decorated and the
    methods are not, so XLA fuses the two differently and reassociates the
    floating-point work.  Measured, that is ~4e-16 relative -- round-off, not a
    second implementation, which a tolerance this tight still catches.
    """
    u = jnp.ones(params_small.n_dof, dtype=complex)

    assert bool(
      jnp.allclose(
        params_small.kernel.apply(u),
        jax_ls.apply_green_function(params_small, u),
        rtol=1e-13,
        atol=0.0,
      )
    )
    assert bool(
      jnp.allclose(
        params_small.kernel.apply_adjoint(u),
        jax_ls.apply_conj_green_function(params_small, u),
        rtol=1e-13,
        atol=0.0,
      )
    )

  def test_default_truncation_radius_is_stored(self):
    """The resolved L is kept, not recomputed by callers."""
    domain = Domain(1.0, 1.0, 8, 8)
    kernel = GreenKernel.create(domain, 10.0)

    assert kernel.truncation_radius == pytest.approx(
      DEFAULT_TRUNCATION_FACTOR * domain.diameter
    )
    assert kernel.truncation_radius > domain.diameter

  def test_explicit_truncation_radius_is_honoured(self):
    domain = Domain(1.0, 1.0, 8, 8)
    kernel = GreenKernel.create(domain, 10.0, truncation_radius=2.0)

    assert kernel.truncation_radius == pytest.approx(2.0)
    # A different L is a different kernel, not a cosmetic label.
    assert not bool(
      jnp.allclose(kernel.ghat, GreenKernel.create(domain, 10.0).ghat)
    )


class TestLippSchwinParams:
  """`LippSchwinParams` is a composition, not a container of copies."""

  def test_init_params_matches_explicit_construction(self):
    """The equivalence `init_params` documents."""
    domain = Domain(1.0, 0.5, 16, 8)
    explicit = jax_ls.LippSchwinParams(
      kernel=GreenKernel.create(domain, 12.0), omega=12.0
    )
    built = jax_ls.init_params(1.0, 0.5, 16, 8, 12.0)

    assert built.domain == explicit.domain
    assert built.omega == explicit.omega
    assert bool(jnp.all(built.GFFT == explicit.GFFT))

  def test_geometry_reads_through_to_the_domain(self):
    """There must be no second, independently stored copy of the spacing.

    Storing `hx` alongside the grid is exactly how the two drifted apart before
    (`code_study.md` §5.2), so this asserts structurally that it cannot recur.
    """
    params = jax_ls.init_params(1.0, 0.5, 16, 8, 12.0)

    for name in ("hx", "hy", "shape", "n_dof", "cell_area"):
      assert name not in params.__dataclass_fields__, (
        f"{name} is a stored field again; it must read through to the domain"
      )
      assert getattr(params, name) == getattr(params.domain, name)

  def test_pytree_holds_only_the_kernel_and_the_frequency(self):
    """Two leaves: `ghat` and `omega`. `X`/`Y` are derived, not carried."""
    params = jax_ls.init_params(1.0, 1.0, 8, 8, 10.0)
    leaves = jax.tree.leaves(params)

    assert len(leaves) == 2
    assert sorted(np.asarray(leaf).size for leaf in leaves) == [1, (4 * 8) ** 2]
