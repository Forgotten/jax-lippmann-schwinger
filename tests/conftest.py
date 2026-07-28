"""Shared pytest fixtures for the jax-lippmann-schwinger test suite."""

import os
import sys

# Ensure the package is importable without installation.  This has to happen
# before `jax_ls` is imported, not after.
sys.path.insert(
  0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import jax
import jax.numpy as jnp
import pytest
from jax import jit, random

import jax_ls

# Enable 64-bit precision (complex128 / float64) before any JAX computation.
jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
AX = 1.0
AY = 1.0
N_SMALL = 2**6  # 64  – used for faster tests
N_LARGE = 2**7  # 128 – used when higher resolution is needed
SAMPLING_RADIUS = 1.0

# A rectangular domain, to exercise the ax != ay / n != m paths.
RECT_AX, RECT_AY = 1.0, 0.5
RECT_N, RECT_M = 2**6, 2**5
RECT_OMEGA = 60.0


# ---------------------------------------------------------------------------
# Fixtures: Lippmann-Schwinger parameters
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def params_small():
  """LippSchwinParams at 64×64 resolution, omega=128."""
  return jax_ls.init_params(AX, AY, N_SMALL, N_SMALL, 128.0)


@pytest.fixture(scope="session")
def params_large():
  """LippSchwinParams at 128×128 resolution, 8 points per wavelength."""
  n = N_LARGE
  omega = 2 * jnp.pi * (n // 8)
  return jax_ls.init_params(AX, AY, n, n, omega)


@pytest.fixture(scope="session")
def params_double():
  """LippSchwinParams at 128×128 on a 2×2 domain, omega=128.

  Chosen so that the grid spacing matches `params_small` exactly.
  """
  return jax_ls.init_params(2 * AX, 2 * AY, 2 * N_SMALL, 2 * N_SMALL, 128.0)


@pytest.fixture(scope="session")
def rect_config():
  """Geometry of the rectangular test domain."""
  return {
    "ax": RECT_AX,
    "ay": RECT_AY,
    "n": RECT_N,
    "m": RECT_M,
    "omega": RECT_OMEGA,
  }


@pytest.fixture(scope="session")
def params_rect():
  """LippSchwinParams on a rectangular domain with n != m and ax != ay."""
  return jax_ls.init_params(RECT_AX, RECT_AY, RECT_N, RECT_M, RECT_OMEGA)


# ---------------------------------------------------------------------------
# Fixtures: Near-field parameters
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def params_nf_small():
  """NearFieldParams at 64×64, omega=128."""
  n = N_SMALL
  return jax_ls.init_params_near_field(AX, AY, n, n, SAMPLING_RADIUS, n, 128.0)


@pytest.fixture(scope="session")
def params_nf_large():
  """NearFieldParams at 128×128, 8 points per wavelength."""
  n = N_LARGE
  omega = 2 * jnp.pi * (n // 8)
  return jax_ls.init_params_near_field(AX, AY, n, n, SAMPLING_RADIUS, n, omega)


@pytest.fixture(scope="session")
def params_nf_born():
  """NearFieldParams at 128×128, 6 points per wavelength (born test)."""
  n = N_LARGE
  omega = 2 * jnp.pi * (n // 6)
  return jax_ls.init_params_near_field(AX, AY, n, n, SAMPLING_RADIUS, n, omega)


@pytest.fixture(scope="session")
def params_nf_rect():
  """NearFieldParams on a rectangular domain with n != m and ax != ay."""
  return jax_ls.init_params_near_field(
    RECT_AX, RECT_AY, RECT_N, RECT_M, SAMPLING_RADIUS, RECT_M, RECT_OMEGA
  )


# ---------------------------------------------------------------------------
# Fixtures: perturbation helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def perturbation_fn():
  """A Gaussian perturbation function."""

  @jit
  def _perturbation(x, y):
    return 1.0 * jnp.exp(-100 * (jnp.square(x) + jnp.square(y)))

  return _perturbation


@pytest.fixture(scope="session")
def delta_perturbation_fn():
  """A narrow Gaussian perturbation (for linearisation tests)."""

  @jit
  def _delta_perturbation(x, y):
    return 0.1 * jnp.exp(-1000 * (jnp.square(x - 0.1) + jnp.square(y - 0.1)))

  return _delta_perturbation


@pytest.fixture(scope="session")
def shifted_perturbation_fn():
  """A Gaussian perturbation shifted from the origin."""

  @jit
  def _perturbation(x, y):
    return 1.0 * jnp.exp(-100 * (jnp.square(x + 0.1) + jnp.square(y + 0.1)))

  return _perturbation


# ---------------------------------------------------------------------------
# Fixtures: reference quadrature
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def direct_volume_potential():
  """Reference volume potential by direct quadrature at a grid point.

  Evaluates ``(G * f)(x_i)`` as ``cell_area * sum_j G(|x_i - x_j|) f(x_j)``
  with the singular self-term punctured.  The evaluation point must be a grid
  point, so that it can be compared against the FFT result without any
  interpolation: comparing an interpolated value against a quadrature at a
  slightly different point hides O(omega * h) discrepancies.
  """
  import numpy as onp
  import scipy as osp

  def _reference(params, f_vect, ix, iy):
    x_e = float(params.X[iy, ix])
    y_e = float(params.Y[iy, ix])
    r = onp.sqrt(
      (onp.asarray(params.X).reshape(-1) - x_e) ** 2
      + (onp.asarray(params.Y).reshape(-1) - y_e) ** 2
    )
    interior = r > 0
    kernel = onp.where(
      interior,
      (1j / 4)
      * osp.special.hankel1(
        0, float(params.omega) * onp.where(interior, r, 1.0)
      ),
      0.0,
    )
    return float(params.cell_area) * onp.sum(kernel * onp.asarray(f_vect))

  return _reference


@pytest.fixture(scope="session")
def nearest_grid_index():
  """Index (ix, iy) of the grid point closest to a requested location."""
  import numpy as onp

  def _nearest(params, x_e, y_e):
    ix = int(onp.argmin(onp.abs(onp.asarray(params.X[0, :]) - x_e)))
    iy = int(onp.argmin(onp.abs(onp.asarray(params.Y[:, 0]) - y_e)))
    return ix, iy

  return _nearest


# ---------------------------------------------------------------------------
# Fixtures: random keys
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def rng_key():
  """A deterministic JAX PRNG key."""
  return random.PRNGKey(0)
