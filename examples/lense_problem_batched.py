import context

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from functools import partial

from jax import grad, jit, vmap
from jax import lax
from jax import random
import jax
import jax.numpy as jnp
import time 

import jax_ls.jax_ls as jax_ls

from jax import random
key = random.PRNGKey(0)

# size of the domain in x and y
ax = 1.0
ay = 1.0

# number of discretization points per dimension
n = 2**8
m = n

# we choose to have 4 points per wavelenght
omega = 2*jnp.pi*(n//4)

# initialize the parameters
params = jax_ls.init_params(ax,ay, n, m, omega)

# definition of the perturbation by the lense
@jit
def perturbation(x,y):
    return jnp.exp(-100*(jnp.square(x-0.5) + jnp.square(y - 0.5)))

# we sample the perturbation
nu = perturbation(params.X, params.Y)
nu_vect  = jnp.reshape(nu, (-1,))


# we choose the direction of the incident field
s = jnp.array([0, 1])
s = s/jnp.linalg.norm(s)  # we normalize it

# we initialize the incident field
u_i = jnp.exp(1j*omega*(params.X*s[0] + params.Y*s[1]))

# we create the right hand side
rhs = -omega**2*nu*u_i

# which is then reshaped
f = jnp.reshape(rhs, (-1,))

# solve for the density u = G*sigma
start = time.time()
sigma = jax_ls.ls_solver(params, nu_vect, f)

# computing the scattered field from the density
u_s = jax_ls.apply_green_function(params, sigma)

end = time.time()
print("overall time elapse was %e[s]"%(end-start))


plt.figure(figsize=(10,10))
plt.subplot(2, 2, 1)
plt.imshow(jnp.real(u_s.reshape(n,m)))
plt.xticks([]); plt.yticks([]);
plt.title('real part', color='red')

plt.subplot(2, 2, 2)
plt.imshow(jnp.imag(u_s.reshape(n,m)))
plt.xticks([]); plt.yticks([]);
plt.title('imaginary part', color='red')

plt.subplot(2, 2, 3)
plt.imshow(jnp.real(u_i + u_s.reshape(n,m)))
plt.xticks([]); plt.yticks([]);

plt.subplot(2, 2, 4)
plt.imshow(jnp.imag(u_i + u_s.reshape(n,m)))
plt.xticks([]); plt.yticks([]);

## now we try to batch and parallelize the solution, which is usefull 
# when computing the farfield operator

n_angles = 10
d_theta = jnp.pi*2/(n_angles)
theta = jnp.linspace(jnp.pi, 3*jnp.pi-d_theta, n_angles)
S = jnp.concatenate([jnp.cos(theta).reshape((n_angles, 1)), 
                     jnp.sin(theta).reshape((n_angles, 1))], axis = 1)

U_i = jnp.exp(1j*omega*(params.X.reshape((-1, 1))*S[:,0].reshape((1, -1))\
                       +params.Y.reshape((-1, 1))*S[:,1].reshape((1, -1))))

Rhs = -omega**2*nu.reshape((-1,1))*U_i

# now we want to solve this problem in a batched form 

solver_jit = jit(partial(jax_ls.ls_solver_batched, params, nu_vect))

solver_batched = jit(vmap(solver_jit, 
                                in_axes=1, 
                                out_axes=1))
green_batched = jit(vmap(partial(jax_ls.apply_green_function, params),
                         in_axes=1,
                         out_axes=1))

# triger compilation with a small righ-hand side
sigma_test = solver_batched(Rhs[:,0:1])

# solve for all the rhs in a vectorized fashion
Sigma = solver_batched(Rhs)

# compue the scattered field
U_s = green_batched(Sigma)

# perhaps do a video here with all the different directions
plt.imshow(jnp.real(U_s[:,1].reshape(n,m)))
plt.show()


# trying a much bigger problem
start = time.time()
n_angles = 100
d_theta = jnp.pi*2/(n_angles)
theta = jnp.linspace(jnp.pi, 3*jnp.pi-d_theta, n_angles)
S = jnp.concatenate([jnp.cos(theta).reshape((n_angles, 1)), 
                     jnp.sin(theta).reshape((n_angles, 1))], axis = 1)

U_i = jnp.exp(1j*omega*(params.X.reshape((-1, 1))*S[:,0].reshape((1, -1))\
                       +params.Y.reshape((-1, 1))*S[:,1].reshape((1, -1))))

Rhs = -omega**2*nu.reshape((-1,1))*U_i

# solve for all the rhs in a vectorized fashion
Sigma = solver_batched(Rhs)

# compue the scattered field
U_s = green_batched(Sigma)

end = time.time()

print("overall time elapse was %d[s]"%(end-start))

