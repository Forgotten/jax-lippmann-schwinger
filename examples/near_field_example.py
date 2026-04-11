import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy as sp
from jax import jit, vmap

import context
import jax_ls

Array = jax.Array

key = jax.random.PRNGKey(0)

# Size of the domain in x and y.
ax = 1.0
ay = 1.0

# Number of discretization points per dimension.
n = 2**7
m = n

# We choose to have 4 points per wavelength.
omega = 2 * jnp.pi * (n // 8)

# Grid spacing.
hx = 1 / (n - 1)

# Initialize the parameters.
params = jax_ls.init_params(ax, ay, n, m, omega)


# Definition of the perturbation by the lens.
@jit
def perturbation(x: Array, y: Array) -> Array:
  return jnp.exp(-100 * (jnp.square(x) + jnp.square(y)))


def green_function(r: Array, omega: float) -> Array:
  return (-1j / 4) * sp.special.hankel1(0, omega * r)


# We sample the perturbation.
nu = perturbation(params.X, params.Y)
nu_vect = jnp.reshape(nu, (-1,))

# Defining the batched transformations.
solver_batched = vmap(
  partial(jax_ls.ls_solver_batched, params, nu_vect),
  in_axes=1,
  out_axes=1,
)

green_batched = vmap(
  partial(jax_ls.apply_green_function, params),
  in_axes=1,
  out_axes=1,
)

# We seek to sample the wavefield on the observation manifold.
start = time.time()
n_angles = n
d_theta = jnp.pi * 2 / (n_angles)
theta = jnp.linspace(jnp.pi, 3 * jnp.pi - d_theta, n_angles)

radius = 1.0

# Defining the observation (and sampling) manifold.
X_s = radius * jnp.concatenate(
  [
    jnp.cos(theta).reshape((n_angles, 1)),
    jnp.sin(theta).reshape((n_angles, 1)),
  ],
  axis=1,
)

# Computing the distances to each evaluation point.
R = jnp.sqrt(
  jnp.square(params.X.reshape((-1, 1)) - X_s[:, 0].reshape((1, -1)))
  + jnp.square(params.Y.reshape((-1, 1)) - X_s[:, 1].reshape((1, -1)))
)

U_i = green_function(R, omega)

Rhs = -(omega**2) * nu.reshape((-1, 1)) * U_i

# Solve for all the RHS in a vectorized fashion.
Sigma = solver_batched(Rhs)

# Compute the scattered field.
U_s = green_batched(Sigma)

end = time.time()
print("overall time elapsed was %d[s]" % (end - start))

near_field = hx**2 * (
  Sigma.T.reshape((n_angles, -1)) @ U_i.reshape(-1, n_angles)
)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(jnp.real(near_field))
plt.xticks([])
plt.yticks([])
plt.title("real part", color="black")

plt.subplot(1, 2, 2)
plt.imshow(jnp.imag(near_field))
plt.xticks([])
plt.yticks([])
plt.title("imaginary part", color="black")

plt.show()

# Using the near-field map.
sampling_radius = 1.0
n_angles = n

params_nf = jax_ls.init_params_near_field(
  ax, ay, n, m, sampling_radius, n_angles, omega
)

start = time.time()

near_field_2 = jax_ls.near_field_map(params_nf, nu_vect)

end = time.time()
print("overall time elapsed was %e[s]" % (end - start))
