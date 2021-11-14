import context

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from functools import partial

from jax import jit

import jax
import jax.numpy as jnp
import time 
import jax.scipy.optimize

import jax_ls

from jax import random
key = random.PRNGKey(0)


# test script to be sure that the adjoint and the linearized version 
# pass the adjoint test

# size of the domain in x and y
ax = 1.0
ay = 1.0

# number of discretization points per dimension
n = 2**7
m = n

# we choose to have 4 points per wavelenght
omega = 2*jnp.pi*(n//8)

# grid spacing
hx = 1/(n-1)

sampling_radious = 1.0
n_angles = n

# initialize the parameters
params_nf = jax_ls.init_params_near_field(ax, ay, n, m,\
                       sampling_radious, n_angles, omega)

# definition of the perturbation by the lense
@jit
def perturbation(x,y):
    return 1.0*jnp.exp(-100*(jnp.square(x) + jnp.square(y)))

@jit
def delta_perturbation(x,y):
    return 0.001*jnp.exp(-1000*(jnp.square(x-0.1) + jnp.square(y-0.1)))

# we sample the perturbation
nu = perturbation(params_nf.ls_params.X, 
                  params_nf.ls_params.Y) 
nu_vect  = jnp.reshape(nu, (-1,))

# we sample the perturbation
delta_nu = delta_perturbation(params_nf.ls_params.X, 
                              params_nf.ls_params.Y) 
delta_nu_vect  = jnp.reshape(delta_nu, (-1,))


# jitting the near field
near_field = jit(partial(jax_ls.near_field_map_vect, params_nf))

# starting the solution and the computation
start = time.time()

# reference wavefield (i.e. data)
near_field_data = near_field(nu_vect + delta_nu_vect)

end = time.time()
print("overall time elapse was %e[s]"%(end-start))

y, J = jax.linearize(near_field, nu_vect)

delta_u_nf = J(delta_nu_vect)

# this is the function with the custom adjoint
near_field_vjp = jit(partial(jax_ls.near_field_map_vect_v2, params_nf))

near_field_data = near_field_vjp(nu_vect)

# obtaining the a
primals, J_star = jax.vjp(near_field_vjp, nu_vect)

rand_cotgnt = jax.random.normal(key, shape=(delta_u_nf.shape[0],)) + 1.j*jax.random.normal(key, shape=(delta_u_nf.shape[0],)) 

# this is not passing the adjoint test... to go back here 
adjoint = J_star(rand_cotgnt)


err = jnp.abs(jnp.sum(jnp.conj(rand_cotgnt)*delta_u_nf)\
      - jnp.sum(jnp.conj(adjoint[0])*delta_nu_vect)*hx**2)\
      /jnp.abs(jnp.sum(jnp.conj(rand_cotgnt)*delta_u_nf))

print("error of adjoint test is %e"%err)

