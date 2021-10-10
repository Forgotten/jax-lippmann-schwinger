import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from jax import grad, jit
from jax import lax
from jax import random
import jax
import jax.numpy as jnp

from jax import random
key = random.PRNGKey(0)

from typing import NamedTuple

def fourier_green_function(L,k,s):
    return (1 + (1j*np.pi/2*L*s)*sp.special.hankel1(0,L*k)*sp.special.jv(1,L*s)\
              - (1j*np.pi/2*L*k)*sp.special.hankel1(1,L*k)*sp.special.jv(0,L*s)\
                           )/(np.square(s) - np.square(k))

# nu_vect = nu(X,Y);
# nu = nu_vect(:);

class LippSchwinParams(NamedTuple):
    GFFT: jnp.ndarray
    omega: jnp.float32
    X: jnp.ndarray
    Y: jnp.ndarray

def init_params(ax, ay, m, n, omega):

    hx = ax/(n-1)
    hy = ay/(n-1)

    x = jnp.linspace(0.,a-hx, n) 
    y = jnp.linspace(0.,a-hy, n)

    [X, Y] = jnp.meshgrid(x,y)

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

    # we define the Fourier kernel
    # check if we can use ifftshift here directly 
    GFFT = jnp.fft.ifftshift(fourier_green_function(L, omega, S))

    return LippSchwinParams(GFFT, omega, X, Y)


@jit
def apply_green_function(params: LippSchwinParams,\
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
    Gu = BExt[:n, :m]

    return jnp.reshape(Gu, (-1,))


# this version works!!!
@jit
def apply_green_function_raw(GFFT, u):

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

@jit
def apply_lipp_schwin(params, nu_vect, u):

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


# wrapper for gmres
@jit
def ls_solver(params, nu_vect, f):
    u, info = jax.scipy.sparse.linalg.gmres(lambda x: apply_lipp_schwin(params,\
                                            nu_vect, x), f )
    return u
