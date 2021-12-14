import context

import numpy as np
import scipy as sp

import matplotlib

import matplotlib.pyplot as plt
from functools import partial

from jax import jit

import jax
import jax.numpy as jnp
import time 
import jax.scipy.optimize

import jax_ls


# size of the domain in x and y
ax = 1.0
ay = 1.0

# number of discretization points per dimension
n = 2**6
m = n

# we choose to have 4 points per wavelenght
omega = 2*jnp.pi*(n//8)

# grid spacing
hx = 1/(n-1)

sampling_radious = 1.0
n_angles = n

# initialize the parameters
params_nf = jax_ls.init_params_near_field(ax, ay, n, m,\
                                         sampling_radious,\
                                         n_angles, omega)

# definition of the perturbation by the lense
@jit
def perturbation(x,y):
    return 1.0*jnp.exp(-500*(jnp.square(x+0.1) + jnp.square(y+0.2)))\
         + 1.0*jnp.exp(-500*(jnp.square(x-0.1) + jnp.square(y-0.1)))


# we sample the perturbation
nu = perturbation(params_nf.ls_params.X, params_nf.ls_params.Y) 
nu_vect  = jnp.reshape(nu, (-1,))


# jitting the near field map (vectorized) with the custom vjp
near_field_vjp = jit(partial(jax_ls.near_field_map_vect_vjp, params_nf))

# reference wavefield (i.e. data)
near_field_data_v2 = near_field_vjp(nu_vect)

# jitting the near field map (vectorized) with the custom vjp
loss_vjp = jit(partial(jax_ls.near_field_l2_loss, params_nf, near_field_data_v2.reshape((m,m))))

print("initial loss with zero initial guess %e"%(loss_vjp(0*nu_vect)))

opt_result = jax.scipy.optimize.minimize(loss_vjp, x0 = jnp.real(0*nu_vect), method = "bfgs")

opt_nu = opt_result.x

print("Final loss with zero initial guess %e"%(loss_vjp(opt_nu)))


# ploting the near field map 
plt.figure(figsize=(15,5))
plt.subplot(1, 3, 1)
plt.imshow(jnp.real(opt_nu).reshape((n,n)))
plt.xticks([]); plt.yticks([]);
plt.title('reconstructed media', color='black')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(jnp.real(nu_vect).reshape((n,n)))
plt.xticks([]); plt.yticks([]);
plt.title('reference media', color='black')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(jnp.abs(nu_vect-opt_nu).reshape((n,n)))
plt.xticks([]); plt.yticks([]);
plt.title('error', color='black')
plt.colorbar()

plt.show()

plt.savefig("nu_recontruction.png")
