import context

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from functools import partial

from jax import jit

import jax
import jax.numpy as jnp
import time 

import jax_ls

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
    return jnp.exp(-100*(jnp.square(x) + jnp.square(y)))

# we sample the perturbation
nu = perturbation(params_nf.ls_params.X, 
                  params_nf.ls_params.Y)
nu_vect = jnp.reshape(nu, (-1,))
 
# starting the solution and the computation
start = time.time()

near_field = jax_ls.near_field_map(params_nf, nu_vect)

end = time.time()
print("overall time elapse was %e[s]"%(end-start))

# ploting the near field map 
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(jnp.real(near_field))
plt.xticks([]); plt.yticks([]);
plt.title('real part', color='black')

plt.subplot(1, 2, 2)
plt.imshow(jnp.imag(near_field))
plt.xticks([]); plt.yticks([]);
plt.title('imaginary part', color='black')

plt.show()
