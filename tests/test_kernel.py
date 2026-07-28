"""Tests for the truncated Green's function in the Fourier domain.

The kernel

  Ghat_L(s) = N(s) / (s^2 - omega^2)

has a removable singularity at |s| = omega: the numerator vanishes there too.
The mode set of the padded grid contains |s| = omega exactly whenever omega is
chosen as a fixed number of points per wavelength, so evaluating the quotient
naively computes 0/0 for most of the frequencies this package is used with.
These tests pin the closed-form limit that replaces it.
"""

import numpy as np
import pytest
import scipy as sp

import jax
import jax.numpy as jnp

import jax_ls

jax.config.update("jax_enable_x64", True)


def _lhopital_limit(L, k):
  """Reference value of Ghat_L at |s| = k, independent of the implementation."""
  return (1j * np.pi * L**2 / 4) * (
    sp.special.hankel1(0, L * k) * sp.special.jv(0, L * k)
    + sp.special.hankel1(1, L * k) * sp.special.jv(1, L * k)
  )


class TestRemovableSingularity:
  """The kernel must be finite and continuous through |s| = omega."""

  @pytest.mark.parametrize("L,k", [(1.5, 16 * np.pi), (1.5, 50.0), (6.0, 10.0)])
  def test_numerator_vanishes(self, L, k):
    """N(k) = 0, which is what makes the singularity removable."""
    numerator = (
      1
      + (1j * np.pi / 2 * L * k)
      * sp.special.hankel1(0, L * k)
      * sp.special.jv(1, L * k)
      - (1j * np.pi / 2 * L * k)
      * sp.special.hankel1(1, L * k)
      * sp.special.jv(0, L * k)
    )
    assert abs(numerator) < 1e-12, (
      f"|N(k)| = {abs(numerator):.2e} should vanish"
    )

  @pytest.mark.parametrize("L,k", [(1.5, 16 * np.pi), (1.5, 50.0), (6.0, 10.0)])
  def test_value_at_singularity_is_finite(self, L, k):
    """Evaluating exactly at |s| = k must not produce inf or nan."""
    value = jax_ls.fourier_green_function(L, k, np.array([k]))
    assert np.isfinite(value).all(), f"Ghat_L({k}) = {value} is not finite"

  @pytest.mark.parametrize("L,k", [(1.5, 16 * np.pi), (1.5, 50.0), (6.0, 10.0)])
  def test_value_at_singularity_matches_limit(self, L, k):
    """The value at |s| = k must be the L'Hopital limit."""
    value = complex(jax_ls.fourier_green_function(L, k, np.array([k]))[0])
    expected = _lhopital_limit(L, k)
    err = abs(value - expected) / abs(expected)
    assert err < 1e-12, f"value at the singularity is off by {err:.2e}"

  @pytest.mark.parametrize("L,k", [(1.5, 16 * np.pi), (6.0, 10.0)])
  def test_continuous_through_the_singularity(self, L, k):
    """The kernel must not jump across |s| = k."""
    # Straddle the singularity, staying outside the tolerance band so that the
    # quotient branch is exercised on both sides.
    eps = 1e-6
    s = np.array([k * (1 - eps), k, k * (1 + eps)])
    values = jax_ls.fourier_green_function(L, k, s)

    assert np.isfinite(values).all()
    for neighbour in (values[0], values[2]):
      err = abs(neighbour - values[1]) / abs(values[1])
      assert err < 1e-3, f"kernel jumps by {err:.2e} across the singularity"

  def test_quotient_branch_still_used_away_from_the_singularity(self):
    """Far from |s| = k the kernel must equal the plain quotient."""
    L, k = 1.5, 50.0
    s = np.linspace(0.5 * k, 0.9 * k, 17)
    numerator = (
      1
      + (1j * np.pi / 2 * L * s)
      * sp.special.hankel1(0, L * k)
      * sp.special.jv(1, L * s)
      - (1j * np.pi / 2 * L * k)
      * sp.special.hankel1(1, L * k)
      * sp.special.jv(0, L * s)
    )
    expected = numerator / (np.square(s) - np.square(k))
    got = jax_ls.fourier_green_function(L, k, s)
    err = float(np.max(np.abs(got - expected) / np.abs(expected)))
    assert err < 1e-13, f"quotient branch disagrees by {err:.2e}"


class TestKernelOnTheGrid:
  """The discrete kernel built by `init_params` must be usable as-is."""

  # Every one of these lands a Fourier mode exactly on |s| = omega.
  POINTS_PER_WAVELENGTH_CONFIGS = [
    (2**6, 8),
    (2**7, 8),
    (2**7, 6),
    (2**8, 6),
    (2**8, 4),
  ]

  @pytest.mark.parametrize("n,ppw", POINTS_PER_WAVELENGTH_CONFIGS)
  def test_gfft_is_finite(self, n, ppw):
    """GFFT must be finite at the usual points-per-wavelength frequencies."""
    omega = 2 * np.pi * (n // ppw)
    params = jax_ls.init_params(1.0, 1.0, n, n, omega)
    ghat = np.asarray(params.GFFT)
    assert np.isfinite(ghat).all(), (
      f"GFFT has {np.count_nonzero(~np.isfinite(ghat))} non-finite entries "
      f"at n={n}, omega=2*pi*{n // ppw}"
    )

  @pytest.mark.parametrize("n,ppw", POINTS_PER_WAVELENGTH_CONFIGS)
  def test_gfft_is_bounded(self, n, ppw):
    """A silently huge entry is as bad as an inf; bound the kernel."""
    omega = 2 * np.pi * (n // ppw)
    params = jax_ls.init_params(1.0, 1.0, n, n, omega)
    # The kernel decays like 1/|s|^2 away from the singularity; its peak is the
    # value at the singularity, which is O(L^2).
    assert float(jnp.max(jnp.abs(params.GFFT))) < 1.0

  def test_kernel_depends_only_on_the_modulus(self):
    """Ghat_L is radial, so GFFT must be symmetric under s -> -s."""
    params = jax_ls.init_params(1.0, 1.0, 2**5, 2**5, 40.0)
    ghat = jnp.fft.fftshift(params.GFFT)
    # fftshift puts the zero mode at index N//2; drop the unpaired first row
    # and column before comparing against the reversed array.
    core = ghat[1:, 1:]
    err = float(jnp.max(jnp.abs(core - core[::-1, ::-1])))
    assert err < 1e-12, f"kernel is not radial, asymmetry {err:.2e}"
