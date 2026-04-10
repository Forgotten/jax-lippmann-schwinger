"""Shared pytest fixtures for the jax-lippmann-schwinger test suite."""

import os
import sys

# Ensure the package is importable without installation.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import jax
# Enable 64-bit precision (complex128 / float64) before any JAX computation.
jax.config.update("jax_enable_x64", True)

import pytest
import jax.numpy as jnp
from jax import jit, random
from functools import partial

import jax_ls


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
AX = 1.0
AY = 1.0
N_SMALL = 2**6   # 64  – used for faster tests
N_LARGE = 2**7   # 128 – used when higher resolution is needed
SAMPLING_RADIUS = 1.0


# ---------------------------------------------------------------------------
# Fixtures: Lippmann-Schwinger parameters
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def params_small():
    """LippSchwinParams at 64×64 resolution, omega=128."""
    n = N_SMALL
    omega = 128.0
    return jax_ls.init_params(AX, AY, n, n, omega)


@pytest.fixture(scope="session")
def params_large():
    """LippSchwinParams at 128×128 resolution, omega from 4 pts/wavelength."""
    n = N_LARGE
    omega = 2 * jnp.pi * (n // 8)
    return jax_ls.init_params(AX, AY, n, n, omega)


@pytest.fixture(scope="session")
def params_double():
    """LippSchwinParams at 128×128 on a 2×2 domain, omega=128."""
    n = N_SMALL
    omega = 128.0
    return jax_ls.init_params(2 * AX, 2 * AY, 2 * n, 2 * n, omega)


# ---------------------------------------------------------------------------
# Fixtures: Near-field parameters
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def params_nf_small():
    """NearFieldParams at 64×64, omega=128."""
    n = N_SMALL
    omega = 128.0
    n_angles = n
    return jax_ls.init_params_near_field(
        AX, AY, n, n, SAMPLING_RADIUS, n_angles, omega
    )


@pytest.fixture(scope="session")
def params_nf_large():
    """NearFieldParams at 128×128, omega from 4 pts/wavelength."""
    n = N_LARGE
    omega = 2 * jnp.pi * (n // 8)
    n_angles = n
    return jax_ls.init_params_near_field(
        AX, AY, n, n, SAMPLING_RADIUS, n_angles, omega
    )


@pytest.fixture(scope="session")
def params_nf_born():
    """NearFieldParams at 128×128, omega from 6 pts/wavelength (born test)."""
    n = N_LARGE
    omega = 2 * jnp.pi * (n // 6)
    n_angles = n
    return jax_ls.init_params_near_field(
        AX, AY, n, n, SAMPLING_RADIUS, n_angles, omega
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
        return 0.1 * jnp.exp(
            -1000 * (jnp.square(x - 0.1) + jnp.square(y - 0.1))
        )
    return _delta_perturbation


@pytest.fixture(scope="session")
def shifted_perturbation_fn():
    """A Gaussian perturbation shifted from the origin."""
    @jit
    def _perturbation(x, y):
        return 1.0 * jnp.exp(
            -100 * (jnp.square(x + 0.1) + jnp.square(y + 0.1))
        )
    return _perturbation


# ---------------------------------------------------------------------------
# Fixtures: random keys
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def rng_key():
    """A deterministic JAX PRNG key."""
    return random.PRNGKey(0)
