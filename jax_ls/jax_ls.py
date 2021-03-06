import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from functools import partial

from jax import grad, jit
from jax import lax
from jax import random
from jax import custom_jvp


import jax
import jax.numpy as jnp


from typing import NamedTuple

def fourier_green_function(L,k,s):
    return (1 + (1j*np.pi/2*L*s)*sp.special.hankel1(0,L*k)*sp.special.jv(1,L*s)\
              - (1j*np.pi/2*L*k)*sp.special.hankel1(1,L*k)*sp.special.jv(0,L*s)\
                           )/(np.square(s) - np.square(k))

# nu_vect = nu(X,Y);
# nu = nu_vect(:);

class LippSchwinParams(NamedTuple):
    # truncated Green's function in Fourier domain
    GFFT: jnp.ndarray
    # frequency
    omega: jnp.float32
    # grid in x
    X: jnp.ndarray
    # grid in y 
    Y: jnp.ndarray


def init_params(ax, ay, n, m, omega):
    """ funciton to initialize the parameters
    ax:    length of the domain in the x direction
    ay:    length of the domian in the y direction
    n:     number of discretization points in the x direction
    m:     number of discretization points in the y direction
    omega: frequency 
    """

    hx = ax/(n-1)
    hy = ay/(m-1)

    x = jnp.linspace(0.,ax-hx, n) - ax/2
    y = jnp.linspace(0.,ay-hy, n) - ay/2

    [X, Y] = jnp.meshgrid(x,y)

    # to do: this should be the largest 
    max_length = jnp.max(jnp.array([ax, ay]))
    Lp = 4*max_length 
    L  = 1.5*max_length

    kx = jnp.linspace(-2*n, 2*n-1, 4*n);
    ky = jnp.linspace(-2*m, 2*m-1, 4*m);

    # to check afterwards (in the case the domain is not 
    # squared)
    # KX = (2*pi/Lp)*repmat(kx', 1, 4*m);
    # KY = (2*pi/Lp)*repmat(ky, 4*n,1); 

    (Kx, Ky) = jnp.meshgrid(kx, ky)
    Kx, Ky = (2*np.pi/Lp)*Kx, (2*np.pi/Lp)*Ky

    S = jnp.sqrt(jnp.square(Kx) + jnp.square(Ky))

    # we define the Fourier kernel
    # check if we can use ifftshift here directly 
    GFFT = jnp.fft.ifftshift(fourier_green_function(L, omega, S))

    return LippSchwinParams(GFFT, omega, X, Y)


@jit
def apply_green_function(params: LippSchwinParams,\
                         u: jnp.ndarray) -> jnp.ndarray:

    # we extract the dimensions
    ne, me = params.GFFT.shape[0], params.GFFT.shape[1]
    n, m =  ne//4, me//4

    # we create the intermediate vector
    BExt = jnp.zeros((ne, me), dtype=np.complex64)
    BExt = BExt.at[:n,:m].set(jnp.reshape(u, (n,m)))

    # Fourier Transform
    BFft = jnp.fft.fft2(BExt);
    # Component-wise multiplication
    BFft = params.GFFT*BFft;
    # Inverse Fourier Transform
    BExt = jnp.fft.ifft2(BFft);

    # we extract the correct piece
    Gu = BExt[:n, :m]

    # we return the reshaped vector
    return jnp.reshape(Gu, (-1,))

@jit
def apply_lipp_schwin(params: LippSchwinParams, 
                      nu_vect: jnp.ndarray,
                      u: jnp.ndarray) -> jnp.ndarray:

    ne, me = params.GFFT.shape[0], params.GFFT.shape[1]
    n, m =  ne//4, me//4

    BExt = jnp.zeros((ne, me), dtype=np.complex64)
    BExt = BExt.at[:n,:m].set(jnp.reshape(u, (n,m)))

    # Fourier Transform
    BFft = jnp.fft.fft2(BExt);
    # Component-wise multiplication
    BFft = params.GFFT*BFft;
    # Inverse Fourier Transform
    BExt = jnp.fft.ifft2(BFft);

    # we extract the correct piece
    Gu = BExt[:n, :m].reshape((-1,))

    # we return the (-I + omega^2 nu G)u 
    return -u + jnp.square(params.omega)*nu_vect*Gu


@jit
def apply_conj_green_function(params: LippSchwinParams,\
                         u: jnp.ndarray) -> jnp.ndarray:
    # function to apply the conjugate of the Green's function
    # this is usefull when computing the adjoint

    # we extract the dimensions
    ne, me = params.GFFT.shape[0], params.GFFT.shape[1]
    n, m =  ne//4, me//4

    # we create the intermediate vector
    BExt = jnp.zeros((ne, me), dtype=np.complex64)
    BExt = BExt.at[:n,:m].set(jnp.reshape(u, (n,m)))

    # Fourier Transform
    BFft = jnp.fft.fft2(BExt);
    # Component-wise multiplication
    BFft = jnp.conj(params.GFFT)*BFft;
    # Inverse Fourier Transform
    BExt = jnp.fft.ifft2(BFft);

    # we extract the correct piece
    Gu = BExt[:n, :m]

    # we return the reshaped vector
    return jnp.reshape(Gu, (-1,))

@jit
def apply_lipp_schwin_adj(params: LippSchwinParams, 
                          nu_vect: jnp.ndarray,
                          u: jnp.ndarray) -> jnp.ndarray:
    # function to apply the conjugate of the Green's function
    
    ne, me = params.GFFT.shape[0], params.GFFT.shape[1]
    n, m =  ne//4, me//4

    BExt = jnp.zeros((ne, me), dtype=np.complex64)
    BExt = BExt.at[:n,:m].set(jnp.reshape(u, (n,m)))

    # Fourier Transform
    BFft = jnp.fft.fft2(BExt);
    # Component-wise multiplication
    BFft = jnp.conj(params.GFFT)*BFft;
    # Inverse Fourier Transform
    BExt = jnp.fft.ifft2(BFft);

    # we extract the correct piece
    Gu = BExt[:n, :m].reshape((-1,))

    # we return the (-I + omega^2 nu G)u 
    return -u + jnp.square(params.omega)*nu_vect*Gu

# wrapper for gmres
@jit
def ls_solver(params: LippSchwinParams, 
              nu_vect: jnp.ndarray,
              f: jnp.ndarray) -> jnp.ndarray:

    sigma, info = jax.scipy.sparse.linalg.gmres(lambda x: apply_lipp_schwin(params,\
                                            nu_vect, x), f )
    return sigma

@jit
def ls_solver_batched(params: LippSchwinParams, 
                      nu_vect: jnp.ndarray,
                      f: jnp.ndarray) -> jnp.ndarray:
    """ 
    Function to solve the density for the Lippmann-Schwinger equation 
    """
    sigma, info = jax.scipy.sparse.linalg.gmres(lambda x: apply_lipp_schwin(params,\
                                            nu_vect, x), f , solve_method='batched')
    return sigma

# TODO there should be an easier way to write this very similar functions
@jit
def ls_solver_batched_adj(params: LippSchwinParams, 
                      nu_vect: jnp.ndarray,
                      f: jnp.ndarray) -> jnp.ndarray:
    """ 
    Function to solve the density for adjoint Lippmann-Schwinger equation 
    """
    sigma, info = jax.scipy.sparse.linalg.gmres(lambda x: apply_lipp_schwin_adj(params,\
                                            nu_vect, x), f , solve_method='batched')
    return sigma


# we compute the Born approximation to compute u directly instead of sigma
@partial(custom_jvp, nondiff_argnums=(0,2))
def ls_solver_u(params: LippSchwinParams, 
                nu_vect: jnp.ndarray,
                f: jnp.ndarray) -> jnp.ndarray:
    # this is basically the same function as above, the difference is that we compute
    # u directly

    # we solve for sigma first (this function is already jitted)
    sigma = ls_solver(params, nu_vect, f)
    
    # we compute u in this case from sigma
    u = apply_green_function(params, sigma)

    return u

@ls_solver_u.defjvp
def ls_solver_u_jvp(params, f, primals, tangents):
    """ Function to compute the Born approximation 
    Lap ( u + delta u) + omega^2 (nu + delta nu) ( u + delta u) = f
    
    we first compute the zero-th order 
    Lap u + omega^2 nu u = f
    which is then used to compute the first variation
    Lap delta_u + omega^2 nu (delta_u) = -omega^2 (delta_nu) u
    
    """

    nu_vect, = primals
    delta_nu_vect, = tangents

    # solve the zero-th order with the reference nu
    u = ls_solver_u(params, nu_vect, f)

    # compute the rhs for the Born approximation
    rhs = -params.omega**2*u*delta_nu_vect

    # solve the equation of the first variation 
    # Lap delta_u + omega^2 nu (delta_u) = -omega^2 (delta_nu) u
    delta_u = ls_solver_u(params, nu_vect, rhs)

    return u, delta_u


# Born approximation only for sigma
@partial(custom_jvp, nondiff_argnums=(0,2))
def ls_solver_batched_sigma(params: LippSchwinParams, 
                            nu_vect: jnp.ndarray,
                            f: jnp.ndarray) -> jnp.ndarray:
    """ 
    Function to solve the density for the Lippmann-Schwinger equation 
    """
    sigma, info = jax.scipy.sparse.linalg.gmres(lambda x: apply_lipp_schwin(params,\
                                            nu_vect, x), f , solve_method='batched')
    return sigma


@ls_solver_batched_sigma.defjvp
def ls_solver_batched_sigma_jvp(params, f, primals, tangents):
    """ Function to compute the Born approximation 
    Lap ( u + delta u) + omega^2 (nu + delta nu) ( u + delta u) = f
    
    we first compute the zero-th order 
    Lap u + omega^2 nu u = f
    which is then used to compute the first variation
    Lap delta u + omega^2 nu (delta u) = -omega^2 (delta nu) u
    """

    nu_vect, = primals
    delta_nu_vect, = tangents


    # solve the zero-th order with the reference nu
    sigma, info = jax.scipy.sparse.linalg.gmres(lambda x: apply_lipp_schwin(params,\
                                            nu_vect, x), f , solve_method='batched')

    u = apply_green_function(params, sigma)

    # compute the rhs for the Born approximation
    rhs = -params.omega**2*u*delta_nu_vect

    delta_sigma, info = jax.scipy.sparse.linalg.gmres(lambda x: apply_lipp_schwin(params,\
                                                    nu_vect, x), rhs)

    return sigma, delta_sigma


#####################################################################
# here we implement a few test functions

# this version works!!!
@jit
def apply_green_function_raw(GFFT: jnp.ndarray, 
                             u: jnp.ndarray) -> jnp.ndarray:

    n, m =  GFFT.shape[0]//4, GFFT.shape[1]//4

    BExt = jnp.zeros(GFFT.shape, dtype=np.complex64)
    BExt = BExt.at[:n,:m].set(jnp.reshape(u,(n,m)))

    # Fourier Transform
    BFft = jnp.fft.fft2(BExt);
    # Component-wise multiplication
    BFft = GFFT*BFft;
    # Inverse Fourier Transform
    BExt = jnp.fft.ifft2(BFft);

    # we extract the correct piece
    Gu = BExt[:n, :m]

    return jnp.reshape(Gu, (-1,))
