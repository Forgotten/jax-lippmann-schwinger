import scipy as sp
from functools import partial

import jax
from jax import jit, vmap
from jax import custom_jvp
from jax import custom_vjp
import jax.numpy as jnp

from flax import struct

Array = jax.Array

from .jax_ls import init_params
from .jax_ls import LippSchwinParams

# Importing the application of the Green's functions and the operators.
from .jax_ls import apply_green_function
from .jax_ls import apply_conj_green_function
# from .jax_ls import apply_lipp_schwin
# from .jax_ls import apply_lipp_schwin_adj

# Importing the different solvers.
from .jax_ls import ls_solver_batched
from .jax_ls import ls_solver_batched_sigma
from .jax_ls import ls_solver_batched_adj

Array = jax.Array


def green_function(r: Array, omega: float) -> Array:
  """Evaluate the fundamental solution of the Helmholtz equation.

  Args:
    r: Distance between points.
    omega: Angular frequency.

  Returns:
    The value of the Green's function at distance r.
  """
  # This is Green's function defined as the fundamental solution.
  # Note the sign, i.e., it satisfies:
  # Delta G(x,y) + omega^2 G(x,y) = - delta (x,y).
  return (1j / 4) * sp.special.hankel1(0, omega * r)


@struct.dataclass
class NearFieldParams:
  """Configuration for the near-field map and its derivatives.

  Attributes:
    ls_params: Lippmann-Schwinger configuration.
    G_sample: Discretized Green's function for the near-field manifold.
    hx: Grid spacing in the x-coordinate.
    hy: Grid spacing in the y-coordinate.
  """

  # Truncated Green's function in Fourier domain.
  ls_params: LippSchwinParams
  # Green's function in space.
  G_sample: Array
  # Grid spacing for the quadratures.
  hx: jnp.float32
  # Grid spacing for the quadratures.
  hy: jnp.float32


def init_params_near_field(
  ax: float, ay: float, n: int, m: int, r: float, n_theta: int, omega: float
) -> NearFieldParams:
  """Initialize parameters for the near-field measurement system.

  Args:
    ax: Length of the domain in the x direction.
    ay: Length of the domain in the y direction.
    n: Number of discretization points in the x direction.
    m: Number of discretization points in the y direction.
    r: Radius of the observation manifold.
    n_theta: Number of samples/sources on the observation manifold.
    omega: Angular frequency.

  Returns:
    An initialized NearFieldParams object.
  """

  hx = ax / (n - 1)
  hy = ay / (m - 1)

  # Initialize params for LS.
  params = init_params(ax, ay, n, m, omega)

  d_theta = jnp.pi * 2 / (n_theta)
  theta = jnp.linspace(jnp.pi, 3 * jnp.pi - d_theta, n_theta)

  # Defining the observation (and sampling) manifold.
  X_s = r * jnp.concatenate(
    [
      jnp.cos(theta).reshape((n_theta, 1)),
      jnp.sin(theta).reshape((n_theta, 1)),
    ],
    axis=1,
  )

  # Computing the distances to each evaluation point.
  R = jnp.sqrt(
    jnp.square(params.X.reshape((-1, 1)) - X_s[:, 0].reshape((1, -1)))
    + jnp.square(params.Y.reshape((-1, 1)) - X_s[:, 1].reshape((1, -1)))
  )

  # Here we are defining the Green's function using the correct sign.
  # It satisfies Delta G(x,y) + omega^2 G(x,y) = delta (x,y).
  U_i = -green_function(R, omega)

  return NearFieldParams(params, U_i, hx, hy)


@jit
def near_field_map(params: NearFieldParams, nu_vect: Array) -> Array:
  """Compute the near field on an observation circle.

  Args:
    params: Configuration for the near-field map.
    nu_vect: Contrast vector.

  Returns:
    The computed near field.
  """

  Rhs = (
    -(params.ls_params.omega**2) * nu_vect.reshape((-1, 1)) * params.G_sample
  )

  # Defining the batched transformations.
  solver_batched = jit(
    vmap(
      jit(partial(ls_solver_batched, params.ls_params, nu_vect)),
      in_axes=1,
      out_axes=1,
    )
  )

  # Solve for all the rhs in a vectorized fashion.
  Sigma = solver_batched(Rhs)

  nm, n_angles = params.G_sample.shape

  # Computing the near field by integrating with the Green's function.
  # The sign is due to convolving with 4/i H^{(1)}_0(k).
  near_field = (
    -(Sigma.T.reshape((n_angles, nm)) @ params.G_sample.reshape((nm, n_angles)))
    * params.hx
    * params.hy
  )

  return near_field


@partial(custom_jvp, nondiff_argnums=(0,))
def near_field_map_vect(params: NearFieldParams, nu_vect: Array) -> Array:
  """Compute the near field using a vectorized solver with custom JVP.

  Args:
    params: Configuration for the near-field map.
    nu_vect: Contrast vector.

  Returns:
    The reshaped near field as a flattened vector.
  """

  Rhs = (
    -(params.ls_params.omega**2) * nu_vect.reshape((-1, 1)) * params.G_sample
  )

  # Defining the batched transformations.
  solver_batched = jit(
    vmap(
      jit(partial(ls_solver_batched_sigma, params.ls_params, nu_vect)),
      in_axes=1,
      out_axes=1,
    )
  )

  # Solve for all the rhs in a vectorized fashion.
  Sigma = solver_batched(Rhs)

  nm, n_angles = params.G_sample.shape

  # TODO: Fix this! It is just a matrix-vector multiplication.
  # Computing the near field by integrating with the Green's function.
  # The sign is due to convolving with 4/i H^{(1)}_0(k).
  near_field = (
    -(Sigma.T.reshape((n_angles, nm)) @ params.G_sample.reshape((nm, n_angles)))
    * params.hx
    * params.hy
  )

  return near_field.reshape((-1,))


@near_field_map_vect.defjvp
def near_field_map_vect_jvp(
  params: NearFieldParams, primals: tuple[Array], tangents: tuple[Array]
) -> tuple[Array, Array]:
  """Compute the Jacobian-vector product for the near-field map.

  Args:
    params: Configuration for the near-field map.
    primals: Tuple of primal inputs (nu_vect).
    tangents: Tuple of tangent inputs (delta_nu_vect).

  Returns:
    A tuple of (primal output, tangent output).
  """
  # Function to compute the Born approximation.
  # Lap ( u + delta u) + omega^2 (nu + delta nu) ( u + delta u) = f
  # We first compute the zero-th order: Lap u + omega^2 nu u = f,
  # which is then used to compute the first variation:
  # Lap delta u + omega^2 nu (delta u) = -omega^2 (delta nu) u.

  (nu_vect,) = primals
  (delta_nu_vect,) = tangents

  Rhs = (
    -(params.ls_params.omega**2) * nu_vect.reshape((-1, 1)) * params.G_sample
  )

  # Defining the batched transformations.
  solver_batched = jit(
    vmap(
      jit(partial(ls_solver_batched, params.ls_params, nu_vect)),
      in_axes=1,
      out_axes=1,
    )
  )

  green_batched = jit(
    vmap(
      jit(partial(apply_green_function, params.ls_params)),
      in_axes=1,
      out_axes=1,
    )
  )

  # Solve for all the rhs in a vectorized fashion.
  Sigma = solver_batched(Rhs)

  U_total = green_batched(Sigma) + params.G_sample

  # Compute the rhs for the Born approximation.
  Rhs_pert = (
    -(params.ls_params.omega**2) * delta_nu_vect.reshape((-1, 1)) * U_total
  )

  # Solving for the first variation of the density.
  delta_Sigma = solver_batched(Rhs_pert)

  # Extracting sizes to evaluate the integrals.
  nm, n_angles = params.G_sample.shape

  # Computing the near field by integrating with the Green's function.
  # The sign is due to convolving with 4/i H^{(1)}_0(k).
  near_field = (
    -(Sigma.T.reshape((n_angles, nm)) @ params.G_sample.reshape((nm, n_angles)))
    * params.hx
    * params.hy
  )

  # Computing the near field variation by integrating with the Green's function.
  # The sign is due to convolving with 4/i H^{(1)}_0(k).
  delta_near_field = (
    -(
      delta_Sigma.T.reshape((n_angles, nm))
      @ params.G_sample.reshape((nm, n_angles))
    )
    * params.hx
    * params.hy
  )

  return near_field.reshape((-1,)), delta_near_field.reshape((-1,))


@partial(custom_vjp, nondiff_argnums=(0,))
def near_field_map_vect_vjp(params: NearFieldParams, nu_vect: Array) -> Array:
  """Compute the near field using a vectorized solver with custom VJP.

  Args:
    params: Configuration for the near-field map.
    nu_vect: Contrast vector.

  Returns:
    The reshaped near field as a flattened vector.
  """

  Rhs = (
    -(params.ls_params.omega**2) * nu_vect.reshape((-1, 1)) * params.G_sample
  )

  # Defining the batched transformations.
  solver_batched = jit(
    vmap(
      jit(partial(ls_solver_batched_sigma, params.ls_params, nu_vect)),
      in_axes=1,
      out_axes=1,
    )
  )

  # Solve for all the rhs in a vectorized fashion.
  Sigma = solver_batched(Rhs)

  nm, n_angles = params.G_sample.shape

  # Computing the near field by integrating with the Green's function.
  # The sign is due to convolving with 4/i H^{(1)}_0(k).
  near_field = (
    -(Sigma.T.reshape((n_angles, nm)) @ params.G_sample.reshape((nm, n_angles)))
    * params.hx
    * params.hy
  )

  return near_field.reshape((-1,))


@jit
def near_field_map_vect_vjp_fwd(
  params: NearFieldParams, nu_vect: Array
) -> tuple[Array, tuple[Array, Array]]:
  """Forward pass for the near-field map VJP.

  Args:
    params: Configuration for the near-field map.
    nu_vect: Contrast vector.

  Returns:
    A tuple of (output near field, residuals for backward pass).
  """

  Rhs = (
    -(params.ls_params.omega**2) * nu_vect.reshape((-1, 1)) * params.G_sample
  )

  # Defining the batched transformations.
  solver_batched = jit(
    vmap(
      jit(partial(ls_solver_batched_sigma, params.ls_params, nu_vect)),
      in_axes=1,
      out_axes=1,
    )
  )

  # Solve for all the rhs in a vectorized fashion.
  Sigma = solver_batched(Rhs)

  nm, n_angles = params.G_sample.shape

  # Computing the near field by integrating with the Green's function.
  # The sign is due to convolving with 4/i H^{(1)}_0(k).
  near_field = (
    -(Sigma.T.reshape((n_angles, nm)) @ params.G_sample.reshape((nm, n_angles)))
    * params.hx
    * params.hy
  )

  return near_field.reshape((-1,)), (Sigma, nu_vect)


@jit
def near_field_map_vect_vjp_bwd(
  params: NearFieldParams, fwd_res: tuple[Array, Array], res: Array
) -> tuple[Array]:
  """Backward pass for the near-field map VJP.

  Args:
    params: Configuration for the near-field map.
    fwd_res: Residuals from the forward pass.
    res: Input cotangents.

  Returns:
    The output cotangent with respect to nu_vect.
  """

  (
    Sigma,
    nu_vect,
  ) = fwd_res

  # Defining the batched for the adjoint problem.
  solver_adj_batched = jit(
    vmap(
      jit(partial(ls_solver_batched_adj, params.ls_params, nu_vect)),
      in_axes=1,
      out_axes=1,
    )
  )

  conj_green_batched = jit(
    vmap(
      jit(partial(apply_conj_green_function, params.ls_params)),
      in_axes=1,
      out_axes=1,
    )
  )

  green_batched = jit(
    vmap(
      jit(partial(apply_green_function, params.ls_params)),
      in_axes=1,
      out_axes=1,
    )
  )

  # Computing the scattered field.
  U_s = green_batched(Sigma)

  # Computing the total field.
  U_total = U_s + params.G_sample

  nm, n_angles = params.G_sample.shape

  # Incident field in the adjoint problem.
  U_adj_i = (params.ls_params.omega**2) * (
    jnp.conj(params.G_sample) @ res.reshape((n_angles, n_angles))
  )

  # Assembling the adjoint RHS for the integral equation.
  Rhs_adjoint = (
    -(params.ls_params.omega**2) * U_adj_i * nu_vect.reshape((-1, 1))
  )

  # Solve for all the rhs in a vectorized fashion.
  Sigma_adj = solver_adj_batched(Rhs_adjoint)

  # Computing the scattered field.
  U_adj = conj_green_batched(Sigma_adj) + U_adj_i

  return (-(jnp.sum(jnp.conj(U_total) * U_adj * params.hx**2, axis=1)),)


# we define the application of the adjoit
near_field_map_vect_vjp.defvjp(
  near_field_map_vect_vjp_fwd, near_field_map_vect_vjp_bwd
)


@partial(custom_vjp, nondiff_argnums=(0,))
def near_field_map_vect_jax(params: NearFieldParams, nu_vect: Array) -> Array:
  """Compute the near field following JAX cotangent conventions.

  Args:
    params: Configuration for the near-field map.
    nu_vect: Contrast vector.

  Returns:
    The reshaped near field as a flattened vector.
  """

  Rhs = (
    -(params.ls_params.omega**2) * nu_vect.reshape((-1, 1)) * params.G_sample
  )

  # Defining the batched transformations.
  solver_batched = jit(
    vmap(
      jit(partial(ls_solver_batched_sigma, params.ls_params, nu_vect)),
      in_axes=1,
      out_axes=1,
    )
  )

  # Solve for all the rhs in a vectorized fashion.
  Sigma = solver_batched(Rhs)

  nm, n_angles = params.G_sample.shape

  # Computing the near field by integrating with the Green's function.
  # The sign is due to convolving with 4/i H^{(1)}_0(k).
  near_field = (
    -(Sigma.T.reshape((n_angles, nm)) @ params.G_sample.reshape((nm, n_angles)))
    * params.hx
    * params.hy
  )

  return near_field.reshape((-1,))


@jit
def near_field_map_vect_jax_fwd(
  params: NearFieldParams, nu_vect: Array
) -> tuple[Array, tuple[Array, Array]]:
  """Forward pass for the JAX-style near-field map VJP.

  Args:
    params: Configuration for the near-field map.
    nu_vect: Contrast vector.

  Returns:
    A tuple of (output near field, residuals for backward pass).
  """

  Rhs = (
    -(params.ls_params.omega**2) * nu_vect.reshape((-1, 1)) * params.G_sample
  )

  # Defining the batched transformations.
  solver_batched = jit(
    vmap(
      jit(partial(ls_solver_batched_sigma, params.ls_params, nu_vect)),
      in_axes=1,
      out_axes=1,
    )
  )

  # Solve for all the rhs in a vectorized fashion.
  Sigma = solver_batched(Rhs)

  nm, n_angles = params.G_sample.shape

  # Computing the near field by integrating with the Green's function.
  # The sign is due to convolving with 4/i H^{(1)}_0(k).
  near_field = (
    -(Sigma.T.reshape((n_angles, nm)) @ params.G_sample.reshape((nm, n_angles)))
    * params.hx
    * params.hy
  )

  return near_field.reshape((-1,)), (Sigma, nu_vect)


@jit
def near_field_map_vect_jax_bwd(
  params: NearFieldParams, fwd_res: tuple[Array, Array], res: Array
) -> tuple[Array]:
  """Backward pass for the JAX-style near-field map VJP.

  Args:
    params: Configuration for the near-field map.
    fwd_res: Residuals from the forward pass.
    res: Input cotangents.

  Returns:
    The output cotangent with respect to nu_vect.
  """

  (Sigma, nu_vect) = fwd_res

  # Defining the batched for the adjoint problem.
  solver_adj_batched = vmap(
    partial(ls_solver_batched_adj, params.ls_params, nu_vect),
    in_axes=1,
    out_axes=1,
  )

  conj_green_batched = vmap(
    partial(apply_conj_green_function, params.ls_params),
    in_axes=1,
    out_axes=1,
  )

  green_batched = vmap(
    partial(apply_green_function, params.ls_params),
    in_axes=1,
    out_axes=1,
  )

  # Computing the scattered field.
  U_s = green_batched(Sigma)

  # Computing the total field.
  U_total = U_s + params.G_sample

  nm, n_angles = params.G_sample.shape

  # Incident field in the adjoint problem.
  # Using the definition from JAX that passes res* instead of res.
  U_adj_i = (params.ls_params.omega**2) * (
    jnp.conj(params.G_sample) @ jnp.conj(res).reshape((n_angles, n_angles))
  )

  # Assembling the adjoint RHS for the integral equation.
  Rhs_adjoint = (
    -(params.ls_params.omega**2) * U_adj_i * nu_vect.reshape((-1, 1))
  )

  # Solve for all the rhs in a vectorized fashion.
  Sigma_adj = solver_adj_batched(Rhs_adjoint)

  # Computing the scattered field.
  U_adj = conj_green_batched(Sigma_adj) + U_adj_i

  return (-jnp.conj(jnp.sum(jnp.conj(U_total) * U_adj * params.hx**2, axis=1)),)


# we define the application of the adjoit
near_field_map_vect_jax.defvjp(
  near_field_map_vect_jax_fwd, near_field_map_vect_jax_bwd
)


@partial(custom_vjp, nondiff_argnums=(0, 1))
def near_field_l2_loss(
  params: NearFieldParams, u_data: Array, nu_vect: Array
) -> Array:
  """Compute the L2 loss for the near-field map.

  Args:
    params: Configuration for the near-field map.
    u_data: Reference near-field data.
    nu_vect: Contrast vector.

  Returns:
    The scalar L2 loss value.
  """

  Rhs = (
    -(params.ls_params.omega**2) * nu_vect.reshape((-1, 1)) * params.G_sample
  )

  # Defining the batched transformations.
  solver_batched = jit(
    vmap(
      jit(partial(ls_solver_batched_sigma, params.ls_params, nu_vect)),
      in_axes=1,
      out_axes=1,
    )
  )

  # Solve for all the rhs in a vectorized fashion.
  Sigma = solver_batched(Rhs)

  nm, n_angles = params.G_sample.shape

  near_field = (
    -jnp.sum(
      Sigma.T.reshape((n_angles, nm, 1))
      * params.G_sample.reshape((1, nm, n_angles)),
      axis=1,
    )
    * params.hx
    * params.hy
  )

  misfit = near_field - u_data

  return 0.5 * jnp.sum(jnp.real(misfit * jnp.conj(misfit)))


@jit
def near_field_l2_loss_fwd(
  params: NearFieldParams, u_data: Array, nu_vect: Array
) -> tuple[Array, tuple[Array, Array, Array]]:
  """Forward pass for the L2 loss VJP.

  Args:
    params: Configuration for the near-field map.
    u_data: Reference near-field data.
    nu_vect: Contrast vector.

  Returns:
    A tuple of (loss value, residuals for backward pass).
  """
  # Forward solver to compute the misfit.

  Rhs = (
    -(params.ls_params.omega**2) * nu_vect.reshape((-1, 1)) * params.G_sample
  )

  # Defining the batched transformations.
  solver_batched = jit(
    vmap(
      jit(partial(ls_solver_batched_sigma, params.ls_params, nu_vect)),
      in_axes=1,
      out_axes=1,
    )
  )

  # Solve for all the rhs in a vectorized fashion.
  Sigma = solver_batched(Rhs)

  nm, n_angles = params.G_sample.shape

  near_field = (
    -(Sigma.T.reshape((n_angles, nm)) @ params.G_sample.reshape((nm, n_angles)))
    * params.hx
    * params.hy
  )

  res = near_field - u_data

  return 0.5 * jnp.sum(jnp.real(res * jnp.conj(res))), (Sigma, res, nu_vect)


@jit
def near_field_l2_loss_bwd(
  params: NearFieldParams,
  u_data: Array,
  fwd_res: tuple[Array, Array, Array],
  cotangent: float,
) -> tuple[Array]:
  """Backward pass for the L2 loss VJP.

  Args:
    params: Configuration for the near-field map.
    u_data: Reference near-field data.
    fwd_res: Residuals from the forward pass.
    cotangent: Input cotangent weight.

  Returns:
    The output cotangent with respect to nu_vect.
  """

  # We load the density for the forward pass and the residual from the misfit.
  Sigma, res, nu_vect = fwd_res

  green_batched = vmap(
    jit(partial(apply_green_function, params.ls_params)),
    in_axes=1,
    out_axes=1,
  )

  # Defining the batched for the adjoint problem.
  solver_adj_batched = vmap(
    jit(partial(ls_solver_batched_adj, params.ls_params, nu_vect)),
    in_axes=1,
    out_axes=1,
  )

  conj_green_batched = vmap(
    jit(partial(apply_conj_green_function, params.ls_params)),
    in_axes=1,
    out_axes=1,
  )

  # Computing the scattered field.
  U_s = green_batched(Sigma)

  # Computing the total field.
  U_total = U_s + params.G_sample

  # Extracting sizes to evaluate the integrals.
  nm, n_angles = params.G_sample.shape

  # Incident field in the adjoint problem.
  U_adj_i = (params.ls_params.omega**2) * (jnp.conj(params.G_sample) @ res)

  # Assembling the adjoint RHS for the integral equation.
  Rhs_adjoint = (
    -(params.ls_params.omega**2) * U_adj_i * nu_vect.reshape((-1, 1))
  )

  # Solve for all the rhs in a vectorized fashion.
  Sigma_adj = solver_adj_batched(Rhs_adjoint)

  # Computing the scattered field.
  U_adj = conj_green_batched(Sigma_adj) + U_adj_i

  return (
    -cotangent
    * jnp.sum(jnp.real(U_adj * jnp.conj(U_total)), axis=1)
    * params.hx**2,
  )


# defining the vector-Jacobian product
near_field_l2_loss.defvjp(near_field_l2_loss_fwd, near_field_l2_loss_bwd)


# @near_field_l2_loss.defjvp
def near_field_l2_loss_grad(
  params: NearFieldParams, u_data: Array, nu_vect: Array
) -> tuple[Array, Array]:
  """Compute the gradient of the L2 loss with respect to the contrast.

  Args:
    params: Configuration for the near-field map.
    u_data: Reference near-field data.
    nu_vect: Contrast vector.

  Returns:
    A tuple of (loss value, gradient vector).
  """
  # Function to compute the Born approximation.
  # Lap ( u + delta u) + omega^2 (nu + delta nu) ( u + delta u) = f
  # We first compute the zero-th order: Lap u + omega^2 nu u = f,
  # which is then used to compute the first variation:
  # Lap delta u + omega^2 nu (delta u) = -omega^2 (delta nu) u.

  Rhs = (
    -(params.ls_params.omega**2) * nu_vect.reshape((-1, 1)) * params.G_sample
  )

  # Defining the batched transformations.
  solver_batched = vmap(
    partial(ls_solver_batched, params.ls_params, nu_vect),
    in_axes=1,
    out_axes=1,
  )

  green_batched = vmap(
    partial(apply_green_function, params.ls_params),
    in_axes=1,
    out_axes=1,
  )

  # Defining the batched for the adjoint problem.
  solver_adj_batched = vmap(
    partial(ls_solver_batched_adj, params.ls_params, nu_vect),
    in_axes=1,
    out_axes=1,
  )

  conj_green_batched = vmap(
    partial(apply_conj_green_function, params.ls_params),
    in_axes=1,
    out_axes=1,
  )

  # Solve for all the rhs in a vectorized fashion.
  Sigma = solver_batched(Rhs)

  # Computing the scattered field.
  U_s = green_batched(Sigma)

  # Computing the total field.
  U_total = U_s + params.G_sample

  # Extracting sizes to evaluate the integrals.
  nm, n_angles = params.G_sample.shape

  # TODO: This is just a matrix vector multiplication.
  # It should be (params.G_sample.T)@Sigma.
  near_field = (
    -(Sigma.T.reshape((n_angles, nm)) @ params.G_sample.reshape((nm, n_angles)))
    * params.hx
    * params.hy
  )

  # Computing the residual.
  res = near_field - u_data

  # Incident field for the adjoint problem.
  U_adj_i = (params.ls_params.omega**2) * (jnp.conj(params.G_sample) @ res)

  # Assembling the adjoint RHS for the integral equation.
  Rhs_adjoint = (
    -(params.ls_params.omega**2) * U_adj_i * nu_vect.reshape((-1, 1))
  )

  # Solve for all the rhs in a vectorized fashion.
  Sigma_adj = solver_adj_batched(Rhs_adjoint)

  # Computing the scattered field.
  U_adj = conj_green_batched(Sigma_adj)

  U_adj_total = U_adj + U_adj_i

  return 0.5 * jnp.sum(jnp.real(res * jnp.conj(res))), -jnp.sum(
    jnp.real(U_adj_total * jnp.conj(U_total)) * params.hx**2, axis=1
  )
