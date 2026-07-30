"""Misfits over several frequencies, and how to weight them.
"""

import enum
from typing import Callable

import jax
import jax.numpy as jnp

from flax import struct

from .near_field import NearFieldParams, near_field_l2_loss

Array = jax.Array


@struct.dataclass
class FrequencyDataset:
  """Measured data at one frequency.

  Attributes:
    params: The near-field configuration for this frequency.
    data: The measured near field, of shape ``params.data_shape``.
    noise_level: Standard deviation of the noise, for `Weighting.NOISE`.
  """

  params: NearFieldParams
  data: Array
  noise_level: float = struct.field(pytree_node=False, default=1.0)

  @property
  def omega(self) -> float:
    """The frequency of this dataset."""
    return self.params.ls_params.omega

  @property
  def n_data(self) -> int:
    """Number of measurements."""
    return self.params.n_sources * self.params.n_receivers


class Weighting(enum.Enum):
  """How to combine the per-frequency misfits.

  Attributes:
    UNIFORM: ``w_k = 1``.  Dominated by whichever frequency is loudest.
    DATA_NORM: ``w_k = 1 / ||D_k||^2``.  Each frequency contributes a relative
      misfit of order one.  The default, and the only option needing no tuning.
    POWER: ``w_k = omega_k^{-p}``.  Fragile -- see the module docstring.
    NOISE: ``w_k = 1 / (sigma_k^2 n_k)``.  Use when the noise level is known.
  """

  UNIFORM = "uniform"
  DATA_NORM = "data_norm"
  POWER = "power"
  NOISE = "noise"


def frequency_weights(
  datasets: list[FrequencyDataset],
  policy: Weighting = Weighting.DATA_NORM,
  power: float = 2.0,
) -> Array:
  """The weight of each frequency under `policy`.

  Args:
    datasets: One dataset per frequency.
    policy: Which rule to use.
    power: Exponent for `Weighting.POWER`.

  Returns:
    An array of weights, one per dataset.
  """
  if policy is Weighting.UNIFORM:
    return jnp.ones(len(datasets))
  if policy is Weighting.DATA_NORM:
    return jnp.asarray([1.0 / jnp.sum(jnp.abs(d.data) ** 2) for d in datasets])
  if policy is Weighting.POWER:
    return jnp.asarray([float(d.omega) ** (-power) for d in datasets])
  if policy is Weighting.NOISE:
    return jnp.asarray([1.0 / (d.noise_level**2 * d.n_data) for d in datasets])
  raise ValueError(f"unknown weighting policy {policy!r}")


def wideband_loss(
  datasets: list[FrequencyDataset], weights: Array, nu_vect: Array
) -> Array:
  """The weighted sum of the per-frequency misfits.

  Each term keeps its own ``custom_vjp``, so the gradient is
  ``sum_k w_k grad(loss_k)`` and costs one adjoint solve per frequency -- no new
  differentiation machinery.

  Args:
    datasets: One dataset per frequency.
    weights: One weight per dataset.
    nu_vect: Contrast.

  Returns:
    The scalar weighted misfit.
  """
  total = 0.0
  for dataset, weight in zip(datasets, weights):
    total = total + weight * near_field_l2_loss(
      dataset.params, dataset.data, nu_vect
    )
  return total


def make_wideband_loss(
  datasets: list[FrequencyDataset],
  policy: Weighting = Weighting.DATA_NORM,
  power: float = 2.0,
) -> Callable[[Array], Array]:
  """Build the simultaneous multi-frequency misfit as a function of the contrast.

  Args:
    datasets: One dataset per frequency.
    policy: How to weight the frequencies.
    power: Exponent for `Weighting.POWER`.

  Returns:
    A callable taking the contrast and returning a real scalar.  Under
    `Weighting.DATA_NORM` it evaluates to about ``K / 2`` at a zero contrast, for
    ``K`` frequencies, regardless of the band.
  """
  weights = frequency_weights(datasets, policy, power)

  def loss(nu_vect: Array) -> Array:
    return wideband_loss(datasets, weights, nu_vect)

  return loss
