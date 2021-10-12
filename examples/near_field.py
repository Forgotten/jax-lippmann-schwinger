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
n = 2**7
m = n

# we choose to have 4 points per wavelenght
omega = 2*jnp.pi*(n//4)

# grid spacing
hx = 1/(n-1)

# initialize the parameters
params = jax_ls.init_params(ax,ay, n, m, omega)

# definition of the perturbation by the lense
@jit
def perturbation(x,y):
    return jnp.exp(-100*(jnp.square(x) + jnp.square(y)))


def green_function(r, omega):
    return (-1j/4)*sp.special.hankel1(0,omega*r)

# we sample the perturbation
nu = perturbation(params.X, params.Y)
nu_vect  = jnp.reshape(nu, (-1,))

# defining the batched transformations
solver_batched = jit(vmap(jit(partial(jax_ls.ls_solver_batched, 
                                      params, nu_vect)), 
                          in_axes=1, 
                          out_axes=1))

green_batched = jit(vmap(partial(jax_ls.apply_green_function,
                                 params),
                         in_axes=1,
                         out_axes=1))


# we seek to sample the wavefield on the observation manifold
start = time.time()
n_angles = n
d_theta = jnp.pi*2/(n_angles)
theta = jnp.linspace(jnp.pi, 3*jnp.pi-d_theta, n_angles)
radious = 1.0

# defining the observation (and sampling) manifold
X_s = radious*jnp.concatenate([jnp.cos(theta).reshape((n_angles, 1)),\
                              jnp.sin(theta).reshape((n_angles, 1))],\
                              axis = 1)

# computing the distances to each evaluation poitn
R = jnp.sqrt(  jnp.square(params.X.reshape((-1, 1))
                          - X_s[:,0].reshape((1, -1)))
             + jnp.square(params.Y.reshape((-1, 1))
                          - X_s[:,1].reshape((1, -1))))

U_i = green_function(R, omega)

Rhs = -omega**2*nu.reshape((-1,1))*U_i

# solve for all the rhs in a vectorized fashion
Sigma = solver_batched(Rhs)

# compue the scattered field
U_s = green_batched(Sigma)

end = time.time()
print("overall time elapse was %d[s]"%(end-start))

near_field = jnp.sum(Sigma*U_i, axis=0)*hx
