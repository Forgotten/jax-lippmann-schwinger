import numpy as np
import scipy as sp
from functools import partial

from jax import jit
from jax import custom_jvp

import jax
import jax.numpy as jnp

from flax import struct

Array = jax.Array


def fourier_green_function(L: Array, k: float, s: Array) -> Array:
  """Evaluate the truncated Green's function in the Fourier domain.

  Args:
    L: The truncation length.
    k: The wavenumber (omega).
    s: The Fourier frequency variable.

  Returns:
    The transformed Green's function evaluated at s.
  """
  return (
    1
    + (1j * np.pi / 2 * L * s)
    * sp.special.hankel1(0, L * k)
    * sp.special.jv(1, L * s)
    - (1j * np.pi / 2 * L * k)
    * sp.special.hankel1(1, L * k)
    * sp.special.jv(0, L * s)
  ) / (np.square(s) - np.square(k))


@struct.dataclass
class LippSchwinParams:
  """Parameters for the Lippmann-Schwinger solver.

  Attributes:
    GFFT: The truncated Green's function in the Fourier domain.
    omega: The angular frequency of the wavefield.
    X: The x-coordinates of the computational grid.
    Y: The y-coordinates of the computational grid.
  """

  # Truncated Green's function in Fourier domain.
  GFFT: Array
  # Frequency.
  omega: jnp.float32
  # Grid in x.
  X: Array
  # Grid in y.
  Y: Array


def init_params(
  ax: float, ay: float, n: int, m: int, omega: float
) -> LippSchwinParams:
  """Initialize the parameters for the Lippmann-Schwinger solver.

  Args:
    ax: Length of the domain in the x direction.
    ay: Length of the domain in the y direction.
    n: Number of discretization points in the x direction.
    m: Number of discretization points in the y direction.
    omega: Angular frequency.

  Returns:
    A LippSchwinParams object containing the initialized parameters.
  """

  hx = ax / (n - 1)
  hy = ay / (m - 1)

  x = jnp.linspace(0.0, ax - hx, n) - ax / 2
  y = jnp.linspace(0.0, ay - hy, n) - ay / 2

  [X, Y] = jnp.meshgrid(x, y)

  # To do: this should be the largest.
  max_length = jnp.max(jnp.array([ax, ay]))
  Lp = 4 * max_length
  L = 1.5 * max_length

  kx = jnp.linspace(-2 * n, 2 * n - 1, 4 * n)
  ky = jnp.linspace(-2 * m, 2 * m - 1, 4 * m)

  # To check afterwards (in the case the domain is not squared).
  # KX = (2*pi/Lp)*repmat(kx', 1, 4*m);
  # KY = (2*pi/Lp)*repmat(ky, 4*n,1);

  (Kx, Ky) = jnp.meshgrid(kx, ky)
  Kx, Ky = (2 * np.pi / Lp) * Kx, (2 * np.pi / Lp) * Ky

  S = jnp.sqrt(jnp.square(Kx) + jnp.square(Ky))

  # We define the Fourier kernel.
  # Check if we can use ifftshift here directly.
  GFFT = jnp.fft.ifftshift(fourier_green_function(L, omega, S))

  return LippSchwinParams(GFFT, omega, X, Y)


@jit
def apply_green_function(params: LippSchwinParams, u: Array) -> Array:
  """Apply the Green's function via FFT-based convolution.

  Args:
    params: Configuration parameters for the solver.
    u: Input wavefield as a flattened array.

  Returns:
    The convolved wavefield as a flattened array.
  """

  # We extract the dimensions.
  ne, me = params.GFFT.shape[0], params.GFFT.shape[1]
  n, m = ne // 4, me // 4

  # We create the intermediate vector.
  BExt = jnp.zeros((ne, me), dtype=params.GFFT.dtype)
  BExt = BExt.at[:n, :m].set(jnp.reshape(u, (n, m)))

  # Fourier Transform.
  BFft = jnp.fft.fft2(BExt)
  # Component-wise multiplication.
  BFft = params.GFFT * BFft
  # Inverse Fourier Transform.
  BExt = jnp.fft.ifft2(BFft)

  # We extract the correct piece.
  Gu = BExt[:n, :m]

  # We return the reshaped vector.
  return jnp.reshape(Gu, (-1,))


@jit
def apply_lipp_schwin(
  params: LippSchwinParams, nu_vect: Array, u: Array
) -> Array:
  """Apply the Lippmann-Schwinger operator to a wavefield.

  Args:
    params: Configuration parameters for the solver.
    nu_vect: Contrast/perturbation vector.
    u: Input wavefield.

  Returns:
    The result of (-I + omega^2 * nu * G) * u.
  """

  ne, me = params.GFFT.shape[0], params.GFFT.shape[1]
  n, m = ne // 4, me // 4

  BExt = jnp.zeros((ne, me), dtype=params.GFFT.dtype)
  BExt = BExt.at[:n, :m].set(jnp.reshape(u, (n, m)))

  # Fourier Transform.
  BFft = jnp.fft.fft2(BExt)
  # Component-wise multiplication.
  BFft = params.GFFT * BFft
  # Inverse Fourier Transform.
  BExt = jnp.fft.ifft2(BFft)

  # We extract the correct piece.
  Gu = BExt[:n, :m].reshape((-1,))

  # We return the (-I + omega^2 nu G)u.
  return -u + jnp.square(params.omega) * nu_vect * Gu


@jit
def apply_conj_green_function(params: LippSchwinParams, u: Array) -> Array:
  """Apply the conjugate of the Green's function via FFT.

  Args:
    params: Configuration parameters for the solver.
    u: Input wavefield.

  Returns:
    The result of G* * u.
  """
  # Function to apply the conjugate of the Green's function.
  # This is useful when computing the adjoint.

  # We extract the dimensions.
  ne, me = params.GFFT.shape[0], params.GFFT.shape[1]
  n, m = ne // 4, me // 4

  # We create the intermediate vector.
  BExt = jnp.zeros((ne, me), dtype=params.GFFT.dtype)
  BExt = BExt.at[:n, :m].set(jnp.reshape(u, (n, m)))

  # Fourier Transform.
  BFft = jnp.fft.fft2(BExt)
  # Component-wise multiplication.
  BFft = jnp.conj(params.GFFT) * BFft
  # Inverse Fourier Transform.
  BExt = jnp.fft.ifft2(BFft)

  # We extract the correct piece.
  Gu = BExt[:n, :m]

  # We return the reshaped vector.
  return jnp.reshape(Gu, (-1,))


@jit
def apply_lipp_schwin_adj(
  params: LippSchwinParams, nu_vect: Array, u: Array
) -> Array:
  """Apply the adjoint Lippmann-Schwinger operator.

  Args:
    params: Configuration parameters for the solver.
    nu_vect: Contrast/perturbation vector.
    u: Input wavefield.

  Returns:
    The result of (-I + omega^2 * conj(G) * nu) * u.
  """
  # Function to apply the conjugate of the Green's function.

  ne, me = params.GFFT.shape[0], params.GFFT.shape[1]
  n, m = ne // 4, me // 4

  BExt = jnp.zeros((ne, me), dtype=params.GFFT.dtype)
  BExt = BExt.at[:n, :m].set(jnp.reshape(u, (n, m)))

  # Fourier Transform.
  BFft = jnp.fft.fft2(BExt)
  # Component-wise multiplication.
  BFft = jnp.conj(params.GFFT) * BFft
  # Inverse Fourier Transform.
  BExt = jnp.fft.ifft2(BFft)

  # We extract the correct piece.
  Gu = BExt[:n, :m].reshape((-1,))

  # We return the (-I + omega^2 nu G)u.
  return -u + jnp.square(params.omega) * nu_vect * Gu


@jit
def ls_solver(params: LippSchwinParams, nu_vect: Array, f: Array) -> Array:
  """Solve the Lippmann-Schwinger equation for the density.

  Args:
    params: Configuration parameters for the solver.
    nu_vect: Contrast/perturbation vector.
    f: Right-hand side vector.

  Returns:
    The solution density sigma.
  """

  sigma, info = jax.scipy.sparse.linalg.gmres(
    lambda x: apply_lipp_schwin(params, nu_vect, x), f
  )
  return sigma


@jit
def ls_solver_batched(
  params: LippSchwinParams, nu_vect: Array, f: Array
) -> Array:
  """Solve the Lippmann-Schwinger equation in batched mode.

  Args:
    params: Configuration parameters for the solver.
    nu_vect: Contrast/perturbation vector.
    f: Right-hand side matrix (multiple right-hand sides).

  Returns:
    The solution density matrix.
  """
  sigma, info = jax.scipy.sparse.linalg.gmres(
    lambda x: apply_lipp_schwin(params, nu_vect, x), f, solve_method="batched"
  )
  return sigma


# TODO: there should be an easier way to write these very similar functions.


@jit
def ls_solver_batched_adj(
  params: LippSchwinParams, nu_vect: Array, f: Array
) -> Array:
  """Solve the adjoint Lippmann-Schwinger equation in batched mode.

  Args:
    params: Configuration parameters for the solver.
    nu_vect: Contrast/perturbation vector.
    f: Right-hand side matrix (multiple right-hand sides).

  Returns:
    The adjoint solution density matrix.
  """
  sigma, info = jax.scipy.sparse.linalg.gmres(
    lambda x: apply_lipp_schwin_adj(params, nu_vect, x),
    f,
    solve_method="batched",
  )
  return sigma


# We compute the Born approximation to compute u directly instead of sigma.
@partial(custom_jvp, nondiff_argnums=(0, 2))
def ls_solver_u(params: LippSchwinParams, nu_vect: Array, f: Array) -> Array:
  """Solve for the wavefield u directly.

  Args:
    params: Configuration parameters for the solver.
    nu_vect: Contrast/perturbation vector.
    f: Right-hand side vector.

  Returns:
    The solution wavefield u.
  """
  # This is basically the same function as above, the difference is that we compute
  # u directly.

  # We solve for sigma first (this function is already jitted).
  sigma = ls_solver(params, nu_vect, f)

  # We compute u in this case from sigma.
  u = apply_green_function(params, sigma)

  return u


@ls_solver_u.defjvp
def ls_solver_u_jvp(
  params: LippSchwinParams,
  f: Array,
  primals: tuple[Array],
  tangents: tuple[Array],
) -> tuple[Array, Array]:
  """Compute the Jacobian-vector product for the wavefield solver.

  Args:
    params: Configuration parameters for the solver.
    f: Right-hand side vector.
    primals: Primal values (nu_vect).
    tangents: Tangent values (delta_nu_vect).

  Returns:
    A tuple containing the primal solution and the tangent solution.
  """
  # Function to compute the Born approximation.
  # Lap ( u + delta u) + omega^2 (nu + delta nu) ( u + delta u) = f
  # we first compute the zero-th order.
  # Lap u + omega^2 nu u = f
  # which is then used to compute the first variation.
  # Lap delta_u + omega^2 nu (delta_u) = -omega^2 (delta_nu) u

  (nu_vect,) = primals
  (delta_nu_vect,) = tangents

  # Solve the zero-th order with the reference nu.
  u = ls_solver_u(params, nu_vect, f)

  # Compute the rhs for the Born approximation.
  rhs = -(params.omega**2) * u * delta_nu_vect

  # Solve the equation of the first variation.
  # Lap delta_u + omega^2 nu (delta_u) = -omega^2 (delta_nu) u
  delta_u = ls_solver_u(params, nu_vect, rhs)

  return u, delta_u


@partial(custom_jvp, nondiff_argnums=(0, 2))
def ls_solver_batched_sigma(
  params: LippSchwinParams, nu_vect: Array, f: Array
) -> Array:
  """Solve for the density sigma in batched mode.

  Args:
    params: Configuration parameters for the solver.
    nu_vect: Contrast/perturbation vector.
    f: Right-hand side matrix.

  Returns:
    The solution density sigma.
  """
  sigma, info = jax.scipy.sparse.linalg.gmres(
    lambda x: apply_lipp_schwin(params, nu_vect, x), f, solve_method="batched"
  )
  return sigma


@ls_solver_batched_sigma.defjvp
def ls_solver_batched_sigma_jvp(
  params: LippSchwinParams,
  f: Array,
  primals: tuple[Array],
  tangents: tuple[Array],
) -> tuple[Array, Array]:
  """Compute the Jacobian-vector product for the batched density solver.

  Function to compute the Born approximation.
  Lap ( u + delta u) + omega^2 (nu + delta nu) ( u + delta u) = f
  We first compute the zero-th order, Lap u + omega^2 nu u = f,
  which is then used to compute the first variation,
  Lap delta u + omega^2 nu (delta u) = -omega^2 (delta nu) u.

  Args:
    params: Configuration parameters for the solver.
    f: Right-hand side matrix.
    primals: Primal values (nu_vect).
    tangents: Tangent values (delta_nu_vect).

  Returns:
    A tuple containing the primal and tangent solution densities.
  """

  (nu_vect,) = primals
  (delta_nu_vect,) = tangents

  # Solve the zero-th order with the reference nu.
  sigma, info = jax.scipy.sparse.linalg.gmres(
    lambda x: apply_lipp_schwin(params, nu_vect, x), f, solve_method="batched"
  )

  u = apply_green_function(params, sigma)

  # Compute the rhs for the Born approximation.
  rhs = -(params.omega**2) * u * delta_nu_vect

  delta_sigma, info = jax.scipy.sparse.linalg.gmres(
    lambda x: apply_lipp_schwin(params, nu_vect, x), rhs
  )

  return sigma, delta_sigma


@jit
def apply_green_function_raw(GFFT: Array, u: Array) -> Array:
  """Apply the Green's function convolution directly.

  Args:
    GFFT: Truncated Green's function in Fourier domain.
    u: Input wavefield.

  Returns:
    The convolved wavefield.
  """

  n, m = GFFT.shape[0] // 4, GFFT.shape[1] // 4

  BExt = jnp.zeros(GFFT.shape, dtype=GFFT.dtype)
  BExt = BExt.at[:n, :m].set(jnp.reshape(u, (n, m)))

  # Fourier Transform.
  BFft = jnp.fft.fft2(BExt)
  # Component-wise multiplication.
  BFft = GFFT * BFft
  # Inverse Fourier Transform.
  BExt = jnp.fft.ifft2(BFft)

  # We extract the correct piece.
  Gu = BExt[:n, :m]

  return jnp.reshape(Gu, (-1,))
