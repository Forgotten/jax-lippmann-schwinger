import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, vmap

import context  # noqa: F401
import jax_ls

Array = jax.Array

key = jax.random.PRNGKey(0)

# Size of the domain in x and y.
ax = 1.0
ay = 1.0

# Number of discretization points per dimension.
n = 2**7
m = n

# We choose to have 8 points per wavelength.
omega = 2 * jnp.pi * (n // 8)

# Grid spacing.
# Initialize the parameters.
params = jax_ls.init_params(ax, ay, n, m, omega)


# Definition of the perturbation by the lens.
@jit
def perturbation(x: Array, y: Array) -> Array:
  return jnp.exp(-100 * (jnp.square(x) + jnp.square(y)))


# We sample the perturbation.
nu_vect = params.domain.sample(perturbation)
nu = params.domain.grid(nu_vect)

# Defining the batched transformations.
solver_batched = vmap(
  partial(jax_ls.ls_solver, params, nu_vect),
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
radius = 1.0

# `SamplingManifold` builds the circle and the sampling kernel, which is minus
# the Green's function between the grid and the observation points.
manifold = jax_ls.SamplingManifold.circle(radius, n_angles)
U_i = manifold.sampling_kernel(params.domain, omega)

Rhs = -(omega**2) * nu.reshape((-1, 1)) * U_i

# Solve for all the RHS in a vectorized fashion.
Sigma = solver_batched(Rhs)

# Compute the scattered field.
U_s = green_batched(Sigma)

end = time.time()
print("overall time elapsed was %d[s]" % (end - start))

near_field = params.cell_area * (
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
