"""Lippmann-Schwinger solver and inverse scattering tools in JAX."""

from .domain import Domain
from .kernels import GreenKernel, green_function
from .solvers import GmresConfig, SolveInfo
from .jax_ls import (
  DEFAULT_PAD,
  DEFAULT_TRUNCATION_FACTOR,
  SINGULARITY_RTOL,
  LippSchwinParams,
  apply_conj_green_function,
  apply_green_function,
  apply_lipp_schwin,
  apply_lipp_schwin_adj,
  fourier_green_function,
  init_params,
  ls_solver,
  ls_solver_adj,
  ls_solver_adj_with_info,
  ls_solver_with_info,
  ls_solver_batched,
  ls_solver_batched_adj,
  ls_solver_batched_sigma,
  ls_solver_field,
  ls_solver_u,
)
from .sampling import (
  IncidentField,
  PlaneWaves,
  PointSources,
  SamplingManifold,
)
from .near_field import (
  NearFieldParams,
  born_adjoint,
  born_linearization,
  init_params_near_field,
  make_near_field_params,
  near_field_l2_loss,
  near_field_l2_loss_and_grad,
  near_field_l2_loss_grad,
  near_field_map,
  near_field_map_forward_mode,
  near_field_map_reverse_mode,
  near_field_map_vect,
  near_field_map_vect_jax,
  near_field_map_vect_vjp,
)

__all__ = [
  # Geometry and kernel.
  "Domain",
  "GreenKernel",
  "IncidentField",
  "PlaneWaves",
  "PointSources",
  "SamplingManifold",
  # Configuration.
  "DEFAULT_PAD",
  "DEFAULT_TRUNCATION_FACTOR",
  "SINGULARITY_RTOL",
  "LippSchwinParams",
  "NearFieldParams",
  "init_params",
  "init_params_near_field",
  "make_near_field_params",
  # Kernels and operators.
  "fourier_green_function",
  "green_function",
  "apply_green_function",
  "apply_conj_green_function",
  "apply_lipp_schwin",
  "apply_lipp_schwin_adj",
  # Solvers.
  "GmresConfig",
  "SolveInfo",
  "ls_solver",
  "ls_solver_adj",
  "ls_solver_with_info",
  "ls_solver_adj_with_info",
  "ls_solver_field",
  # Near-field map and derivatives.
  "near_field_map",
  "near_field_map_forward_mode",
  "near_field_map_reverse_mode",
  "born_linearization",
  "born_adjoint",
  # Misfit.
  "near_field_l2_loss",
  "near_field_l2_loss_and_grad",
  # Deprecated aliases, kept so existing scripts keep working.
  "ls_solver_batched",
  "ls_solver_batched_adj",
  "ls_solver_batched_sigma",
  "ls_solver_u",
  "near_field_l2_loss_grad",
  "near_field_map_vect",
  "near_field_map_vect_jax",
  "near_field_map_vect_vjp",
]
