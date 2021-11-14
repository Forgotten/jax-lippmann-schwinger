import context

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from functools import partial

from jax import grad, jit, vmap
from jax import lax
from jax import random
from jax import jvp

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
n = 2**8
m = n

# we choose to have 6 points per wavelenght
omega = 2*jnp.pi*(n//6)

# initialize the parameters
params = jax_ls.init_params(ax,ay, n, m, omega)

# definition of the perturbation by the lense
@jit
def perturbation(x,y):
    return 1.0*jnp.exp(-100*(jnp.square(x) + jnp.square(y)))

@jit
def delta_perturbation(x,y):
    return 0.01*jnp.exp(-1000*(jnp.square(x-0.1) + jnp.square(y-0.1)))


# we sample the perturbation
nu = perturbation(params.X, params.Y) + 0.j
nu_vect  = jnp.reshape(nu, (-1,))

# we sample the perturbation
delta_nu = delta_perturbation(params.X, params.Y) + 0.j
delta_nu_vect  = jnp.reshape(delta_nu, (-1,))


# we choose the direction of the incident field
s = jnp.array([0, 1])
s = s/jnp.linalg.norm(s)  # we normalize it

# we initialize the incident field
u_i = jnp.exp(1j*omega*(params.X*s[0] + params.Y*s[1]))

# we create the right hand side
rhs = -omega**2*nu*u_i

# which is then reshaped
f = jnp.reshape(rhs, (-1,))

# let's encapsulate this a bit
@jit
def solver_u(nu): 
	return jax_ls.ls_solver_u(params, nu, f)

# Finite difference approximation of the derivatives
delta_t = 0.01
u_p_1 = solver_u(nu_vect + delta_t*delta_nu_vect)
u_m_1 = solver_u(nu_vect - delta_t*delta_nu_vect)

fd_approx = (u_p_1-u_m_1)/(2*delta_t)

# using the Jacobian-vector product
u_0, u_born = jvp(solver_u, (nu_vect,), (delta_nu_vect,))

# computing the error
err = jnp.linalg.norm(fd_approx - u_born)/jnp.linalg.norm(u_born)

print("Error in the approximation of the born approximation is %e"%err)

plt.figure(figsize=(10,10))
plt.subplot(2, 2, 1)
plt.imshow(jnp.real(fd_approx.reshape(n,m)))
plt.xticks([]); plt.yticks([]);
plt.title('real part', color='black')
plt.ylabel('FD approximation field')

plt.subplot(2, 2, 2)
plt.imshow(jnp.imag(fd_approx.reshape(n,m)))
plt.xticks([]); plt.yticks([]);
plt.title('imaginary part', color='black')

plt.subplot(2, 2, 3)
plt.imshow(jnp.real(u_born.reshape(n,m)))
plt.xticks([]); plt.yticks([]);
plt.ylabel('first variation of the Field')

plt.subplot(2, 2, 4)
plt.imshow(jnp.imag(u_born.reshape(n,m)))
plt.xticks([]); plt.yticks([]);

plt.show()
