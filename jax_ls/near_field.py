import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from functools import partial

from jax import grad, jit, vmap
from jax import lax
from jax import custom_jvp
from jax import custom_vjp


from jax import random
import jax
import jax.numpy as jnp


from typing import NamedTuple


from .jax_ls import init_params
from .jax_ls import LippSchwinParams

# importing the application of the Green's functions
# and the operators
from .jax_ls import apply_green_function
from .jax_ls import apply_conj_green_function
from .jax_ls import apply_lipp_schwin
from .jax_ls import apply_lipp_schwin_adj

# importing the different solvers
from .jax_ls import ls_solver_batched
from .jax_ls import ls_solver_batched_sigma
from .jax_ls import ls_solver_batched_adj



def green_function(r, omega):
    # this is green's function defined by as the fundamental solution
    # note the sign, i.e., it satisfies,
    # Delta G(x,y) + omega^2 G(x,y) = - delta (x,y)
    return (1j/4)*sp.special.hankel1(0,omega*r)

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

    # here we are defining the Green's function using the correct sign
    # it, it satisfies Delta G(x,y) + omega^2 G(x,y) = delta (x,y)
    U_i = -green_function(R, omega)

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


########################################################################################
########################################################################################
# implementing the near field from vector to vector and using the born approximation


@partial(custom_jvp, nondiff_argnums=(0,))
def near_field_map_vect(params: NearFieldParams, 
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

    # TODO: fix this! it is just a matrix-vector multiplication
    near_field = jnp.sum( Sigma.T.reshape((n_angles, nm, 1))\
                         *params.G_sample.reshape((1, nm, n_angles)),
                         axis=1)*params.hx*params.hy

    return near_field.reshape((-1,))


@near_field_map_vect.defjvp
def near_field_map_vect_jvp(params, primals, tangents):
    """ Function to compute the Born approximation 
    Lap ( u + delta u) + omega^2 (nu + delta nu) ( u + delta u) = f
    
    we first compute the zero-th order 
    Lap u + omega^2 nu u = f
    which is then used to compute the first variation
    Lap delta u + omega^2 nu (delta u) = -omega^2 (delta nu) u

    We can use this to solve the inverse problem using GMRES, i.e., 
    solving the linearized system.
    """

    nu_vect, = primals
    delta_nu_vect, = tangents

    Rhs = -(params.ls_params.omega**2)\
           *nu_vect.reshape((-1,1))*params.G_sample

    # defining the batched transformations
    solver_batched = jit(vmap(jit(partial(ls_solver_batched,
                                          params.ls_params,
                                          nu_vect)),
                              in_axes=1, 
                              out_axes=1))

    green_batched = jit(vmap(jit(partial(apply_green_function,
                                         params.ls_params)),
                             in_axes=1,
                             out_axes=1))


    # solve for all the rhs in a vectorized fashion
    Sigma = solver_batched(Rhs)

    U_total = green_batched(Sigma) + params.G_sample

    # compute the rhs for the Born approximation
    Rhs_pert = -(params.ls_params.omega**2)\
                *delta_nu_vect.reshape((-1,1))*U_total

    # solving for the first variation of the density 
    delta_Sigma = solver_batched(Rhs_pert)

    # extracting sizes to evaluate the integrals  
    nm, n_angles = params.G_sample.shape

    near_field = jnp.sum( Sigma.T.reshape((n_angles, nm, 1))\
                         *params.G_sample.reshape((1, nm, n_angles)),
                         axis=1)*params.hx*params.hy

    delta_near_field = jnp.sum( delta_Sigma.T.reshape((n_angles, nm, 1))\
                         *params.G_sample.reshape((1, nm, n_angles)),
                         axis=1)*params.hx*params.hy


    return near_field.reshape((-1,)), delta_near_field.reshape((-1,))


########################################################################################
########################################################################################

## adding a near field with custom J^*
# we implemented the application of the gradient using the born approximation

@partial(custom_vjp, nondiff_argnums=(0,))
def near_field_map_vect_v2(params: NearFieldParams, 
                        nu_vect: jnp.ndarray) -> jnp.ndarray:
    """function to comput the near field in a circle of radious 1, 
    in this case we use the ls_solver_batched_sigma, which already has 
    a custom vjp """

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

    return near_field.reshape((-1,))


@jit
def near_field_map_vect_v2_fwd(params: NearFieldParams, 
                               nu_vect: jnp.ndarray):
    
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

    return near_field.reshape((-1,)), (Sigma, nu_vect)


@jit
def near_field_map_vect_v2_bwd(params, fwd_res, res):
    """ Function to compute adjoint of the near field operator
    """

    Sigma, nu_vect = fwd_res

    # defining the batched for the adjoint problem
    solver_adj_batched = jit(vmap(jit(partial(ls_solver_batched_adj,
                                          params.ls_params,
                                          nu_vect)),
                              in_axes=1, 
                              out_axes=1))

    conj_green_batched = jit(vmap(jit(partial(apply_conj_green_function,
                                         params.ls_params)),
                             in_axes=1,
                             out_axes=1))

    green_batched = jit(vmap(jit(partial(apply_green_function,
                                         params.ls_params)),
                             in_axes=1,
                             out_axes=1))


    # computing the scattered field
    U_s = green_batched(Sigma)

    # computing the total field
    U_total = U_s + params.G_sample

    nm, n_angles = params.G_sample.shape

    # incident field in the adjoint problem 
    U_adj_i = (params.ls_params.omega**2)*(jnp.conj(params.G_sample)\
              @res.reshape((n_angles,n_angles)))

    # assembling the adjoint RHS for the integral equation
    Rhs_adjoint = -(params.ls_params.omega**2)*U_adj_i*nu_vect.reshape((-1,1))

    # solve for all the rhs in a vectorized fashion
    Sigma_adj = solver_adj_batched(Rhs_adjoint)

    # computing the scattered field
    U_adj = conj_green_batched(Sigma_adj) + U_adj_i


    return (jnp.sum(jnp.conj(U_total)*U_adj, axis = 1),)


# we define the application of the adjoit
near_field_map_vect_v2.defvjp(near_field_map_vect_v2_fwd, near_field_map_vect_v2_bwd)

########################################################################################
########################################################################################
### We implement the L2 loss function

## implementing the L2 loss
@partial(custom_vjp, nondiff_argnums=(0,1))
def near_field_l2_loss(params: NearFieldParams, 
            u_data: jnp.ndarray,
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

    return 0.5*jnp.sum(jnp.real((near_field - u_data) * jnp.conj(near_field - u_data)))
 

def near_field_l2_loss_fwd(params: NearFieldParams, 
            u_data: jnp.ndarray,
            nu_vect: jnp.ndarray) -> jnp.ndarray:
    # forward solver to compute the misfit

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

    res = near_field - u_data

    return 0.5*jnp.sum(jnp.real(res * jnp.conj(res))), (Sigma, res, nu_vect)

def near_field_l2_loss_bwd(params, u_data, fwd_res, cotangent):

    # we load the density for the forward pass and the residual from the 
    # misfit
    Sigma, res, nu_vect = fwd_res

    green_batched = jit(vmap(jit(partial(apply_green_function,
                                         params.ls_params)),
                             in_axes=1,
                             out_axes=1))

    # defining the batched for the adjoint problem
    solver_adj_batched = jit(vmap(jit(partial(ls_solver_batched_sigma,
                                          params.ls_params,
                                          nu_vect)),
                              in_axes=1, 
                              out_axes=1))

    conj_green_batched = jit(vmap(jit(partial(apply_conj_green_function,
                                         params.ls_params)),
                             in_axes=1,
                             out_axes=1))



    # computing the scattered field
    U_s = green_batched(Sigma)

    # computing the total field
    U_total = U_s + params.G_sample

    # extracting sizes to evaluate the integrals  
    nm, n_angles = params.G_sample.shape

    # incident field in the adjoint problem 
    U_adj_i = (params.ls_params.omega**2)*(jnp.conj(params.G_sample)@res)

    # assembling the adjoint RHS for the integral equation
    Rhs_adjoint = -(params.ls_params.omega**2)*U_adj_i*nu_vect.reshape((-1,1))

    # solve for all the rhs in a vectorized fashion
    Sigma_adj = solver_adj_batched(Rhs_adjoint)

    # computing the scattered field
    U_adj = conj_green_batched(Sigma_adj) + U_adj_i

    return (cotangent*jnp.sum(jnp.real(U_adj*jnp.conj(U_total)), axis = 1),)


near_field_l2_loss.defvjp(near_field_l2_loss_fwd, near_field_l2_loss_bwd)


# @near_field_l2_loss.defjvp
def near_field_l2_loss_grad(params, u_data, nu_vect):
    """ Function to compute the Born approximation 
    Lap ( u + delta u) + omega^2 (nu + delta nu) ( u + delta u) = f
    
    we first compute the zero-th order 
    Lap u + omega^2 nu u = f
    which is then used to compute the first variation
    Lap delta u + omega^2 nu (delta u) = -omega^2 (delta nu) u
    """

    Rhs = -(params.ls_params.omega**2)\
           *nu_vect.reshape((-1,1))*params.G_sample

    # defining the batched transformations
    solver_batched = jit(vmap(jit(partial(ls_solver_batched,
                                          params.ls_params,
                                          nu_vect)),
                              in_axes=1, 
                              out_axes=1))

    green_batched = jit(vmap(jit(partial(apply_green_function,
                                         params.ls_params)),
                             in_axes=1,
                             out_axes=1))

    # defining the batched for the adjoint problem
    solver_adj_batched = jit(vmap(jit(partial(ls_solver_batched_sigma,
                                          params.ls_params,
                                          nu_vect)),
                              in_axes=1, 
                              out_axes=1))

    conj_green_batched = jit(vmap(jit(partial(apply_conj_green_function,
                                         params.ls_params)),
                             in_axes=1,
                             out_axes=1))

    # solve for all the rhs in a vectorized fashion
    Sigma = solver_batched(Rhs)

    # computing the scattered field
    U_s = green_batched(Sigma)

    # computing the total field
    U_total = U_s + params.G_sample

    # extracting sizes to evaluate the integrals  
    nm, n_angles = params.G_sample.shape

    # TODO: this is just a matrix vector multiplication
    # it should be (params.G_sample.T)@Sigma
    near_field = jnp.sum( Sigma.T.reshape((n_angles, nm, 1))\
                         *params.G_sample.reshape((1, nm, n_angles)),
                         axis=1)*params.hx*params.hy

    # computing the residual
    res = near_field - u_data 

    # incident field for the adjoint problem
    U_adj_i = (params.ls_params.omega**2)*(jnp.conj(params.G_sample)@res)

    # assembling the adjoint RHS for the integral equation
    Rhs_adjoint = -(params.ls_params.omega**2)*U_adj_i*nu_vect.reshape((-1,1))

    # solve for all the rhs in a vectorized fashion
    Sigma_adj = solver_adj_batched(Rhs_adjoint)

    # computing the scattered field
    U_adj = conj_green_batched(Sigma_adj)

    U_adj_total = U_adj + U_adj_i

    return 0.5*jnp.sum(jnp.real((near_field - u_data) * jnp.conj(near_field - u_data))), jnp.sum(jnp.real(U_adj_total*jnp.conj(U_total)), axis = 1)
