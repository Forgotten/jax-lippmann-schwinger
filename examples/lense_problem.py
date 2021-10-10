import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from functools import partial

from jax import grad, jit, vmap
from jax import lax
from jax import random
import jax
import jax.numpy as jnp

from jax import random
key = random.PRNGKey(0)

a = 1.0

n = 2**10
# we choose to have 4 points per wavelenght
omega = n//4

m = n
h = a/(n-1)

x = jnp.linspace(0.,1.-h, n) 
[X, Y] = jnp.meshgrid(x,x)

Lp = 4*a ; 
L  = a*1.5;


kx = jnp.linspace(-2*n, 2*n-1, 4*n);
ky = jnp.linspace(-2*m, 2*m-1, 4*m);

# to check
# KX = (2*pi/Lp)*repmat(kx', 1, 4*m);
# KY = (2*pi/Lp)*repmat(ky, 4*n,1); 

(Kx, Ky) = jnp.meshgrid(kx, ky)
Kx, Ky = (2*np.pi/Lp)*Kx, (2*np.pi/Lp)*Ky

S = jnp.sqrt(jnp.square(Kx) + jnp.square(Ky))

GFFT = jnp.fft.ifftshift(fourier_green_function(L, omega, S))

# we define the parameters
params = init_params(1.,1., n, m, omega)

# definition of the perturbation by the lense
def perturbation(x,y):
    return np.exp(-100*(np.square(x-0.5) + np.square(y - 0.5)))


# we sample the perturbation
nu = perturbation(X,Y)
nu_vect  = jnp.reshape(nu, (-1,))

# we create the right hand side
rhs = -omega**2*nu*np.exp(1j*omega*X)

# which is then reshaped
f = jnp.reshape(rhs, (-1,))



# creating a random source (why not?)
x = random.normal(key, shape = (n*m,)) + 1j*random.normal(key, shape = (n*m,))

#trigger the jit compilation

LSx = apply_lipp_schwin(params, nu_vect, x)

u, info = jax.scipy.sparse.linalg.gmres(lambda x: apply_lipp_schwin(params, nu_vect, x), f )

sol = apply_green_function(params, u)

plt.imshow(jnp.real(sol.reshape(n,m)))
plt.show()


u_jit = ls_solver(params, nu_vect, f)

solver_fixed = jit(lambda x: ls_solver(params, nu_vect, x))

solver_fixed_batched = vmap(solver_fixed, in_axes=1, out_axes=1)


rhs_2 = -omega**2*nu*np.exp(1j*omega*Y)

F = jnp.concatenate([f.reshape((-1,1)), rhs_2.reshape((-1,1))], axis = 1)

U = solver_fixed_batched(F)


green_batched = jit(vmap(partial(apply_green_function, params),in_axes=1, out_axes=1))
Sol = green_batched(U)

plt.imshow(jnp.real(Sol[:,1].reshape(n,m)))
plt.show()
