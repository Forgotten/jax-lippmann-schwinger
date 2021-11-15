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

import jax_ls

from jax import random
key = random.PRNGKey(0)

# size of the domain in x and y
ax = 1.0
ay = 1.0

# number of discretization points per dimension
n = 2**6
m = n

# we choose to have 6 points per wavelenght
omega = 128.

# initialize the parameters
params = jax_ls.init_params(ax,ay, n, m, omega)

# initialize the parameters 2
params_2 = jax_ls.init_params(2*ax,2*ay, 2*n, 2*m, omega)

# definition of the perturbation by the lense
@jit
def perturbation(x,y):
    return jnp.exp(-100*(jnp.square(x) + jnp.square(y)))

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

# trigger the compilation 
test = jax_ls.apply_lipp_schwin(params, nu_vect, f)

# solve for the density u = G*sigma
sigma = jax_ls.ls_solver(params, nu_vect, f)

# computing the scattered field from the density
u_s = jax_ls.apply_green_function(params, sigma)


# we sample the perturbation
nu = perturbation(params_2.X, params_2.Y)
nu_vect_2  = jnp.reshape(nu, (-1,))

# we choose the direction of the incident field
s = jnp.array([0, 1])
s = s/jnp.linalg.norm(s)  # we normalize it

# we initialize the incident field
u_2_i = jnp.exp(1j*omega*(params_2.X*s[0] + params_2.Y*s[1]))

# we create the right hand side
rhs = -omega**2*nu*u_2_i

# which is then reshaped
f = jnp.reshape(rhs, (-1,))

# trigger the compilation 
test = jax_ls.apply_lipp_schwin(params_2, nu_vect_2, f)

# solve for the density u = G*sigma
sigma_2 = jax_ls.ls_solver(params_2, nu_vect_2, f)

# computing the scattered field from the density
u_2_s = jax_ls.apply_green_function(params_2, sigma_2)


plt.figure(figsize=(10,10))
plt.subplot(2, 2, 1)
plt.imshow(jnp.real(u_s.reshape(n,m)))
plt.xticks([]); plt.yticks([]);
plt.title('real part', color='black')
plt.ylabel('scattered field')

plt.subplot(2, 2, 2)
plt.imshow(jnp.imag(u_s.reshape(n,m)))
plt.xticks([]); plt.yticks([]);
plt.title('imaginary part', color='black')

plt.subplot(2, 2, 3)
plt.imshow(jnp.real(u_i + u_s.reshape(n,m)))
plt.xticks([]); plt.yticks([]);
plt.ylabel('total field')


plt.subplot(2, 2, 4)
plt.imshow(jnp.imag(u_i + u_s.reshape(n,m)))
plt.xticks([]); plt.yticks([]);

plt.savefig("wavefield_small.png")

plt.figure(figsize=(10,10))
plt.subplot(2, 2, 1)
plt.imshow(jnp.real(u_2_s.reshape(2*n,2*m)))
plt.xticks([]); plt.yticks([]);
plt.title('real part', color='black')
plt.ylabel('scattered field')

plt.subplot(2, 2, 2)
plt.imshow(jnp.imag(u_2_s.reshape(2*n,2*m)))
plt.xticks([]); plt.yticks([]);
plt.title('imaginary part', color='black')

plt.subplot(2, 2, 3)
plt.imshow(jnp.real(u_2_i + u_2_s.reshape(2*n,2*m)))
plt.xticks([]); plt.yticks([]);
plt.ylabel('total field')


plt.subplot(2, 2, 4)
plt.imshow(jnp.imag(u_2_i + u_2_s.reshape(2*n,2*m)))
plt.xticks([]); plt.yticks([]);

plt.savefig("wavefield_large.png")

# plt.show()

# pick a value in the bigger domain

x_eval = params_2.X[-10,:]
y_eval = params_2.Y[-10,:]

R = jnp.sqrt( jnp.square(params.X.reshape((1, -1)) - x_eval.reshape(-1,1))\
            + jnp.square(params.Y.reshape((1, -1)) - y_eval.reshape(-1,1)))

U_i = -jax_ls.green_function(R, omega)

res = -(params.X[0,1] - params.X[0,0])**2*U_i@sigma
res2 = -(1./n)**2*U_i@sigma


ref = u_2_s.reshape((2*n,2*m))[-10,:]

err = jnp.sqrt(jnp.sum(jnp.square(jnp.abs(res - ref))*(params.X[0,1] - params.X[0,0])))

print("error between Fourier application and normal quadrature is %e"%err)


plt.figure(figsize=(10,10))
plt.plot(res)
plt.plot(ref)
plt.savefig("difference_between_discretizations.png")
