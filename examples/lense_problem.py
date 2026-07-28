import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit

import context  # noqa: F401
import jax_ls

key = jax.random.PRNGKey(0)

# Size of the domain in x and y.
ax = 1.0
ay = 1.0

# Number of discretization points per dimension.
n = 2**8
m = n

# We choose to have 6 points per wavelength.
omega = 2 * jnp.pi * (n // 6)

# Initialize the parameters.
params = jax_ls.init_params(ax, ay, n, m, omega)

# Definition of the perturbation by the lens.


@jit
def perturbation(x, y):
  return jnp.exp(-100 * (jnp.square(x) + jnp.square(y)))


# We sample the perturbation.
nu = perturbation(params.X, params.Y)
nu_vect = jnp.reshape(nu, (-1,))

# We choose the direction of the incident field.
s = jnp.array([0, 1])
s = s / jnp.linalg.norm(s)  # We normalize it.

# We initialize the incident field.
u_i = jnp.exp(1j * omega * (params.X * s[0] + params.Y * s[1]))

# We create the right hand side.
rhs = -(omega**2) * nu * u_i

# Which is then reshaped.
f = jnp.reshape(rhs, (-1,))

# Trigger the compilation.
test = jax_ls.apply_lipp_schwin(params, nu_vect, f)

# Solve for the density u = G * sigma.
sigma = jax_ls.ls_solver(params, nu_vect, f)

# Computing the scattered field from the density.
u_s = jax_ls.apply_green_function(params, sigma)

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.imshow(jnp.real(u_s.reshape(n, m)))
plt.xticks([])
plt.yticks([])
plt.title("real part", color="black")
plt.ylabel("scattered field")

plt.subplot(2, 2, 2)
plt.imshow(jnp.imag(u_s.reshape(n, m)))
plt.xticks([])
plt.yticks([])
plt.title("imaginary part", color="black")

plt.subplot(2, 2, 3)
plt.imshow(jnp.real(u_i + u_s.reshape(n, m)))
plt.xticks([])
plt.yticks([])
plt.ylabel("total field")

plt.subplot(2, 2, 4)
plt.imshow(jnp.imag(u_i + u_s.reshape(n, m)))
plt.xticks([])
plt.yticks([])

plt.show()
