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
n = 2**7
m = n

hx = 1/(n-1)

# we choose to have 6 points per wavelenght
omega = 2*jnp.pi*(n//6)

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

# we choose the direction of the incident field
x_s = jnp.array([0, 1])

# we initialize the incident field
u_i = green_function(jnp.sqrt(  jnp.square(params.X-x_s[0])\
                              + jnp.square(params.Y-x_s[1])), omega)

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

plt.show()

# sampling on a circle 

# we seek to sample the wavefield on the observation manifold
start = time.time()
n_angles = n
d_theta = jnp.pi*2/(n_angles)
theta = jnp.linspace(jnp.pi, 3*jnp.pi-d_theta, n_angles)
radious = 1.0
X_s = radious*jnp.concatenate([jnp.cos(theta).reshape((n_angles, 1)), 
                       jnp.sin(theta).reshape((n_angles, 1))], axis = 1)

R = jnp.sqrt(  jnp.square(params.X.reshape((-1, 1))-X_s[:,0].reshape((1, -1)))\
             + jnp.square(params.Y.reshape((-1, 1))-X_s[:,1].reshape((1, -1))))

U_i = green_function(R, omega)

# trapezoidal rule 
u_s_bdy = jnp.sum(U_i*sigma.reshape((-1,1)), axis = 0)*(hx**2) 

plt.plot(jnp.real(u_s_bdy))
plt.show()