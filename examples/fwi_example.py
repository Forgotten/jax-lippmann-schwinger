import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.optimize
import matplotlib.pyplot as plt
from jax import jit

import context  # noqa: F401
import jax_ls

# Size of the domain in x and y.
ax = 1.0
ay = 1.0

# Number of discretization points per dimension.
n = 2**6
m = n

# We choose to have 8 points per wavelength.
omega = 2 * jnp.pi * (n // 8)

sampling_radius = 1.0
n_angles = n

# Initialize the parameters.
params_nf = jax_ls.init_params_near_field(
  ax, ay, n, m, sampling_radius, n_angles, omega
)


# Definition of the perturbation by the lens.
@jit
def perturbation(x, y):
  return 1.0 * jnp.exp(
    -500 * (jnp.square(x + 0.1) + jnp.square(y + 0.2))
  ) + 1.0 * jnp.exp(-500 * (jnp.square(x - 0.1) + jnp.square(y - 0.1)))


# We sample the perturbation.
nu = perturbation(params_nf.ls_params.X, params_nf.ls_params.Y)
nu_vect = jnp.reshape(nu, (-1,))

# Jitting the near field map (vectorized) with the custom VJP.
near_field_vjp = jit(partial(jax_ls.near_field_map_reverse_mode, params_nf))

# Reference wavefield (i.e., data).
near_field_data_v2 = near_field_vjp(nu_vect)

# Jitting the near field map (vectorized) with the custom VJP.
loss_vjp = jit(
  partial(
    jax_ls.near_field_l2_loss, params_nf, near_field_data_v2.reshape((m, m))
  )
)

print("initial loss with zero initial guess %e" % (loss_vjp(0 * nu_vect)))

start_time = time.perf_counter()
opt_result = jax.scipy.optimize.minimize(
  loss_vjp, x0=jnp.real(0 * nu_vect), method="bfgs"
)
end_time = time.perf_counter()

opt_nu = opt_result.x

print("Final loss with zero initial guess %e" % (loss_vjp(opt_nu)))
print("Time elapsed in the optimization loop %.2f[s]" % (end_time - start_time))

# Plotting the near field map.
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(jnp.real(opt_nu).reshape((n, n)))
plt.xticks([])
plt.yticks([])
plt.title("reconstructed media", color="black")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(jnp.real(nu_vect).reshape((n, n)))
plt.xticks([])
plt.yticks([])
plt.title("reference media", color="black")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(jnp.abs(nu_vect - opt_nu).reshape((n, n)))
plt.xticks([])
plt.yticks([])
plt.title("error", color="black")
plt.colorbar()

plt.show()

plt.savefig("nu_reconstruction.png")
