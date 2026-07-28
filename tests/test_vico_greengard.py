"""Section 5.1 of Vico, Greengard & Ferrando, arXiv:1604.03155.

The paper validates the truncated-kernel FFT convolution by applying it to a
compactly supported Gaussian source, for which the volume potential is known in
closed form, and reporting the error against that exact solution as the grid is
refined (their Figure 3).  The claim is *superalgebraic* convergence.

This file implements the two-dimensional Helmholtz case, which is exactly what
`apply_green_function` computes.  With

  rho(r) = 1 / (2 pi sigma^2) * exp(-r^2 / (2 sigma^2))            (their Eq. 18)

the volume potential is radial, and expanding the free-space Green's function
`g_k(r) = (i/4) H_0^{(1)}(k r)` in the addition theorem
`(1/2pi) int g_k(|x - y|) dtheta = (i/4) J_0(k r_<) H_0^{(1)}(k r_>)` gives

  [g_k * rho](r) = (i pi / 2) [ H_0^{(1)}(k r) int_0^r J_0(k y) rho(y) y dy
                              + J_0(k r)       int_r^inf H_0^{(1)}(k y) rho(y) y dy ]

which is their Eq. 20.  The two radial integrals are evaluated here by adaptive
quadrature to ~1e-15, so the reference is exact for the purposes of the test.

Note that the source is centred, so the singularity of the kernel sits *inside*
its support and the error is measured over the whole grid, including there.
That is the point of the method, and it is what a plain quadrature rule cannot
do -- the other integration tests in this suite deliberately evaluate away from
the source.

Their Section 5.2 (variable-coefficient Poisson-Boltzmann) is 3-D and out of
scope here.  Section 5.3 (Lippmann-Schwinger) reports *self*-convergence rather
than an exact error and uses a Bi-CGStab tolerance of 1e-12; our GMRES runs at
its 1e-5 default, which would floor any such measurement, so that test is worth
adding only once the solver tolerance is configurable.
"""

import warnings

import numpy as np
import pytest
from scipy import integrate, special

import jax
import jax.numpy as jnp

import jax_ls

jax.config.update("jax_enable_x64", True)

# Parameters from the paper (Figure 3): unit box, sigma = 0.05, k = 2.
SIGMA = 0.05
K_PAPER = 2.0

# The Fourier modes of the padded grid are spaced by 2*pi / (pad * a) = pi / 2,
# so this wavenumber lands exactly on one.  Running the same convergence study
# there exercises the removable singularity of the kernel (see test_kernel.py).
K_ON_MODE = 2 * np.pi

# The reference is computed once on this grid; the coarser grids used in the
# convergence study are exact subsets of it, since h = 1 / n halves cleanly.
REF_N = 2**7
COARSE_N = [2**5, 2**6, 2**7]

# The Gaussian is below 1e-80 of its peak beyond this radius.
SUPPORT_RADIUS = 20 * SIGMA


def _gaussian(r):
  """The source density of Eq. 18, normalised to unit mass."""
  return np.exp(-np.square(r) / (2 * SIGMA**2)) / (2 * np.pi * SIGMA**2)


def _exact_potential(r_grid, k):
  """Evaluate Eq. 20 on a grid of radii.

  The two radial integrals are accumulated over the sorted unique radii, so each
  adaptive quadrature covers one short interval and the whole grid costs O(#radii)
  quadratures rather than O(#radii) full integrations.
  """
  radii = np.unique(r_grid)

  def integrand_j(t):
    return special.j0(k * t) * _gaussian(t) * t

  def integrand_h(t):
    return special.hankel1(0, k * t) * _gaussian(t) * t

  def quad(fn, a, b):
    return integrate.quad(fn, a, b, epsabs=1e-16, epsrel=1e-13, limit=200)[0]

  with warnings.catch_warnings():
    # The Hankel function has an integrable log singularity at the origin; the
    # quadrature handles it but reports a roundoff warning while doing so.
    warnings.simplefilter("ignore", integrate.IntegrationWarning)

    # inner[i] = int_0^{radii[i]} J_0(k y) rho(y) y dy
    edges = np.concatenate([[0.0], radii])
    inner = np.cumsum(
      [quad(integrand_j, edges[i], edges[i + 1]) for i in range(len(radii))]
    )

    # outer[i] = int_{radii[i]}^inf H_0(k y) rho(y) y dy, accumulated downwards
    edges = np.concatenate([radii, [SUPPORT_RADIUS]])
    pieces = np.array(
      [
        quad(lambda t: integrand_h(t).real, edges[i], edges[i + 1])
        + 1j * quad(lambda t: integrand_h(t).imag, edges[i], edges[i + 1])
        for i in range(len(radii))
      ]
    )
    outer = np.cumsum(pieces[::-1])[::-1]

  hankel = np.where(
    radii > 0, special.hankel1(0, k * np.maximum(radii, 1e-300)), 0.0
  )
  potential = (1j * np.pi / 2) * (
    hankel * inner + special.j0(k * radii) * outer
  )

  return potential[np.searchsorted(radii, r_grid)]


def _relative_errors(k, n, reference):
  """Relative L2 and Linf error of the FFT convolution against Eq. 20."""
  params = jax_ls.init_params(1.0, 1.0, n, n, k)
  radii = np.hypot(np.asarray(params.X), np.asarray(params.Y))

  source = jnp.asarray(_gaussian(radii).reshape(-1) + 0j)
  computed = np.asarray(jax_ls.apply_green_function(params, source)).reshape(
    n, n
  )

  exact = reference[:: REF_N // n, :: REF_N // n]
  assert exact.shape == computed.shape

  difference = computed - exact
  return (
    np.linalg.norm(difference) / np.linalg.norm(exact),
    np.max(np.abs(difference)) / np.max(np.abs(exact)),
  )


@pytest.fixture(scope="module")
def reference_paper():
  """Eq. 20 on the finest grid, at the paper's k = 2."""
  params = jax_ls.init_params(1.0, 1.0, REF_N, REF_N, K_PAPER)
  radii = np.hypot(np.asarray(params.X), np.asarray(params.Y))
  return _exact_potential(radii, K_PAPER)


@pytest.fixture(scope="module")
def reference_on_mode():
  """Eq. 20 on the finest grid, at a wavenumber sitting on a Fourier mode."""
  params = jax_ls.init_params(1.0, 1.0, REF_N, REF_N, K_ON_MODE)
  radii = np.hypot(np.asarray(params.X), np.asarray(params.Y))
  return _exact_potential(radii, K_ON_MODE)


class TestGaussianSource:
  """Section 5.1: convolution with a Gaussian source, 2-D Helmholtz."""

  # Once the Gaussian is resolved the trapezoid rule is spectrally accurate and
  # the only remaining error is round-off in the FFTs and in the reference.
  CONVERGED_TOL = 1e-12

  def test_reference_matches_a_direct_quadrature(
    self, reference_paper, direct_volume_potential, nearest_grid_index
  ):
    """Cross-check Eq. 20 against a 2-D quadrature at an exterior point.

    Guards the reference itself: if the radial reduction or the sign convention
    were wrong, every other assertion in this file would be wrong with it.  The
    evaluation point is far outside the support of rho (which is below 1e-16 of
    its peak there), so a punctured trapezoid sum is accurate.
    """
    n = REF_N
    params = jax_ls.init_params(1.0, 1.0, n, n, K_PAPER)
    radii = np.hypot(np.asarray(params.X), np.asarray(params.Y))
    source = _gaussian(radii).reshape(-1) + 0j

    ix, iy = nearest_grid_index(params, 0.35, 0.30)
    direct = direct_volume_potential(params, source, ix, iy)

    err = abs(direct - reference_paper[iy, ix]) / abs(direct)
    assert err < 1e-10, f"Eq. 20 vs direct quadrature: {err:.2e}"

  @pytest.mark.parametrize("n", COARSE_N)
  def test_error_against_the_exact_solution(self, n, reference_paper):
    """Report the error of the FFT convolution at each resolution."""
    l2, linf = _relative_errors(K_PAPER, n, reference_paper)
    # n = 32 puts only ~1.6 points per sigma, so it is not yet resolved; the
    # tight statement is made by the two tests below.
    assert l2 < 1e-7, f"n={n}: relative L2 error {l2:.3e}"
    assert np.isfinite(linf)

  def test_spectral_accuracy_once_resolved(self, reference_paper):
    """Beyond ~3 points per sigma the error is at round-off."""
    for n in (2**6, 2**7):
      l2, linf = _relative_errors(K_PAPER, n, reference_paper)
      assert l2 < self.CONVERGED_TOL, (
        f"n={n}: relative L2 error {l2:.3e} exceeds {self.CONVERGED_TOL:.0e}"
      )
      assert linf < self.CONVERGED_TOL, (
        f"n={n}: relative Linf error {linf:.3e} exceeds {self.CONVERGED_TOL:.0e}"
      )

  def test_superalgebraic_convergence(self, reference_paper):
    """Doubling the grid must gain far more than any algebraic rate allows.

    Measured, the error drops from ~4e-9 to ~1e-15 between n=32 and n=64 -- six
    orders of magnitude for one refinement, where a p-th order method would gain
    2^p.  This is the claim of Figure 3.
    """
    coarse, _ = _relative_errors(K_PAPER, 2**5, reference_paper)
    fine, _ = _relative_errors(K_PAPER, 2**6, reference_paper)

    gain = coarse / fine
    assert gain > 1e4, (
      f"error went from {coarse:.3e} to {fine:.3e}, a factor of {gain:.1e}; "
      "spectral convergence should gain far more than an algebraic rate"
    )


class TestGaussianSourceOnAFourierMode:
  """The same study at a wavenumber that lands on a Fourier mode.

  The kernel's removable singularity is evaluated by its closed-form limit
  there; this checks that doing so does not cost any accuracy.
  """

  def test_spectral_accuracy_once_resolved(self, reference_on_mode):
    for n in (2**6, 2**7):
      l2, _ = _relative_errors(K_ON_MODE, n, reference_on_mode)
      assert l2 < 1e-12, f"n={n}: relative L2 error {l2:.3e}"

  def test_superalgebraic_convergence(self, reference_on_mode):
    coarse, _ = _relative_errors(K_ON_MODE, 2**5, reference_on_mode)
    fine, _ = _relative_errors(K_ON_MODE, 2**6, reference_on_mode)
    assert coarse / fine > 1e4, (
      f"error went from {coarse:.3e} to {fine:.3e} at a wavenumber sitting on "
      "a Fourier mode"
    )
