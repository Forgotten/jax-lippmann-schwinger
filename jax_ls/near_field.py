import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from jax import grad, jit
from jax import lax
from jax import random
import jax
import jax.numpy as jnp


from typing import NamedTuple


from jax_ls import init_params
from jax_ls import LippSchwinParams
from jax_ls import apply_green_function
from jax_ls import apply_lipp_schwin
from jax_ls import ls_solver_batched


class NearFielParams(NamedTuple):
    # truncated Green's function in Fourier domain
    ls_params: LippSchwinParams
    # Green's function in space
    G_sample: jnp.ndarray
    # grid spacing for the quadratures
    hx: jnp.float32
    hy: jnp.float32
    

def init_params_near_field(ax, ay, n, m, omega):
    """ funciton to initialize the parameters
    ax:    length of the domain in the x direction
    ay:    length of the domian in the y direction
    n:     number of discretization points in the x direction
    m:     number of discretization points in the y direction
    omega: frequency 
    """

    hx = ax/(n-1)
    hy = ay/(m-1)

    # initilize params for LS
    params = init_params(ax, ay, n, m, omega)

    n_angles = n
	d_theta = jnp.pi*2/(n_angles)
	theta = jnp.linspace(jnp.pi, 3*jnp.pi-d_theta, n_angles)

	# todo: extract this 

	radious = 1.0 # to be extracted 

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

    return NearFielParams(GFFT, U_i, hx, hy)


def near_field(params: NearFieldParams, 
               nu_vect: jnp.ndarray) -> jnp.ndarray:
"""function to comput the near field in a circle of radious 1"""


	Rhs = -(params.ls_params.omega**2)*nu.reshape((-1,1))*params.G_sample

	# defining the batched transformations
	solver_batched = jit(vmap(jit(partial(ls_solver_batched,
	                                      params.ls_params,
	                                      nu_vect)),
		                      in_axes=1, 
		                      out_axes=1))

	# solve for all the rhs in a vectorized fashion
	Sigma = solver_batched(Rhs)

	near_field = jnp.sum(Sigma*params.G_sample, axis=0)*params.hx*params.hy

	return near_field
