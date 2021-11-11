import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from functools import partial

from jax import grad, jit, vmap
from jax import lax
from jax import random
import jax
import jax.numpy as jnp


from typing import NamedTuple


from .jax_ls import init_params
from .jax_ls import LippSchwinParams
from .jax_ls import apply_green_function
from .jax_ls import apply_lipp_schwin
from .jax_ls import ls_solver_batched


def green_function(r, omega):
    return (-1j/4)*sp.special.hankel1(0,omega*r)

class NearFieldParams(NamedTuple):
    # truncated Green's function in Fourier domain
    ls_params: LippSchwinParams
    # Green's function in space
    G_sample: jnp.ndarray
    # grid spacing for the quadratures
    hx: jnp.float32
    hy: jnp.float32

    
def init_params_near_field(ax, ay, n, m, r, n_theta, omega):
    """ funciton to initialize the parameters
    ax:    length of the domain in the x direction
    ay:    length of the domian in the y direction
    n:     number of discretization points in the x direction
    m:     number of discretization points in the y direction
    r:      radious of the observation manifold
    n_theta: number of samples and point sources in the obs manifold
    omega: frequency 
    """

    hx = ax/(n-1)
    hy = ay/(m-1)

    # initilize params for LS
    params = init_params(ax, ay, n, m, omega)

    d_theta = jnp.pi*2/(n_theta)
    theta = jnp.linspace(jnp.pi, 3*jnp.pi-d_theta, n_theta)

    # defining the observation (and sampling) manifold
    X_s = r*jnp.concatenate([jnp.cos(theta).reshape((n_theta, 1)),\
                                  jnp.sin(theta).reshape((n_theta, 1))],\
                                  axis = 1)

    # computing the distances to each evaluation poitn
    R = jnp.sqrt(  jnp.square(params.X.reshape((-1, 1))
                              - X_s[:,0].reshape((1, -1)))
                 + jnp.square(params.Y.reshape((-1, 1))
                              - X_s[:,1].reshape((1, -1))))

    U_i = green_function(R, omega)

    return NearFieldParams(params, U_i, hx, hy)


@jit
def near_field_map(params: NearFieldParams, 
                   nu_vect: jnp.ndarray) -> jnp.ndarray:
    """function to comput the near field in a circle of radious 1"""

    Rhs = -(params.ls_params.omega**2)\
           *nu_vect.reshape((-1,1))*params.G_sample

    # defining the batched transformations
    solver_batched = jit(vmap(jit(partial(ls_solver_batched,
                                          params.ls_params,
                                          nu_vect)),
                              in_axes=1, 
                              out_axes=1))

    # solve for all the rhs in a vectorized fashion
    Sigma = solver_batched(Rhs)

    nm, n_angles = params.G_sample.shape

    near_field = jnp.sum( Sigma.T.reshape((n_angles, nm, 1))\
                         *params.G_sample.reshape((1, nm, n_angles)),
                         axis=1)*params.hx*params.hy

    return near_field




@jit
def near_field_map_v2(params: NearFieldParams, 
                      nu_vect: jnp.ndarray) -> jnp.ndarray:
    """function to comput the near field in a circle of radious 1, 
    in this case we use the ls_solver_batched_sigma, which already has 
    a custom jvp implemented """

    Rhs = -(params.ls_params.omega**2)\
           *nu_vect.reshape((-1,1))*params.G_sample

    # defining the batched transformations
    solver_batched = jit(vmap(jit(partial(ls_solver_batched_sigma,
                                          params.ls_params,
                                          nu_vect)),
                              in_axes=1, 
                              out_axes=1))

    # solve for all the rhs in a vectorized fashion
    Sigma = solver_batched(Rhs)

    nm, n_angles = params.G_sample.shape

    near_field = jnp.sum( Sigma.T.reshape((n_angles, nm, 1))\
                         *params.G_sample.reshape((1, nm, n_angles)),
                         axis=1)*params.hx*params.hy

    return near_field


