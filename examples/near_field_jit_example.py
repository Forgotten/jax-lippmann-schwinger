import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax

import context
import jax_ls

Array = jax.Array

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

sampling_radius = 1.0
n_angles = n

# Initialize the parameters.
params_nf = jax_ls.init_params_near_field(
  ax, ay, n, m, sampling_radius, n_angles, omega
)


# Definition of the perturbation by the lens.
@jax.jit
def perturbation(x: Array, y: Array) -> Array:
  return jnp.exp(-100 * (jnp.square(x) + jnp.square(y)))


# We sample the perturbation.
nu = perturbation(params_nf.ls_params.X, params_nf.ls_params.Y)
nu_vect = jnp.reshape(nu, (-1,))

# Starting the solution and the computation.
start = time.time()

near_field = jax_ls.near_field_map(params_nf, nu_vect)

end = time.time()
print("overall time elapsed was %e[s]" % (end - start))

# Plotting the near field map.
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
