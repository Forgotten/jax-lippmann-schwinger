"""The computational grid.

`Domain` describes the rectangular region the scatterer lives in and how it is
discretised.  It is deliberately *pure geometry*: it knows nothing about the
frequency, the kernel or the measurement setup, so a single `Domain` can be
shared by every frequency of a wide-band experiment.

Grid convention
---------------
The domain ``[-ax/2, ax/2) x [-ay/2, ay/2)`` is treated as *periodic*: ``n``
(resp. ``m``) points of spacing ``hx = ax / n`` (resp. ``hy = ay / m``), with no
duplicated endpoint.  This is exactly the grid the FFT-based convolution
assumes -- the padded transform has period ``pad * ax`` on ``pad * n`` points --
so the quadrature weight ``hx * hy`` is consistent with the operator.

Every field is static (``pytree_node=False``), so a `Domain` has **no array
leaves**.  That has three consequences worth knowing:

* it can be used as a ``jax.jit`` static argument and costs nothing to carry
  through a trace;
* ``n`` and ``m`` stay concrete, which they must be -- ``jnp.arange(self.n)``
  inside ``jit`` would fail on a traced value, since the output shape would be
  data dependent;
* ``jax.tree.map`` does not touch them, so geometry cannot be silently zeroed by
  a ``zeros_like`` sweep the way a leaf would be.

`X` and `Y` are therefore *recomputed* on access rather than stored.  Inside a
traced function they fold into compile-time constants, and not storing them
means one `Domain` shared across `K` frequencies instead of `K` copies of two
``(m, n)`` arrays.
"""

import numpy as np

import jax
import jax.numpy as jnp

from flax import struct

Array = jax.Array


@struct.dataclass
class Domain:
  """A uniform periodic grid on ``[-ax/2, ax/2) x [-ay/2, ay/2)``.

  Attributes:
    ax: Length of the domain in the x direction.
    ay: Length of the domain in the y direction.
    n: Number of discretization points in the x direction.
    m: Number of discretization points in the y direction.
  """

  ax: float = struct.field(pytree_node=False)
  ay: float = struct.field(pytree_node=False)
  n: int = struct.field(pytree_node=False)
  m: int = struct.field(pytree_node=False)

  def __post_init__(self):
    if self.n < 2 or self.m < 2:
      raise ValueError(
        f"the grid must have at least 2 points per axis, got {(self.n, self.m)}"
      )
    if self.ax <= 0 or self.ay <= 0:
      raise ValueError(f"the domain must have positive size, got {self.size}")

  # -- size ----------------------------------------------------------------
  @property
  def size(self) -> tuple[float, float]:
    """Physical size ``(ax, ay)`` of the domain."""
    return (self.ax, self.ay)

  @property
  def diameter(self) -> float:
    """Largest distance between two points of the domain."""
    return float(np.hypot(self.ax, self.ay))

  # -- discretisation ------------------------------------------------------
  @property
  def hx(self) -> float:
    """Grid spacing in x."""
    return self.ax / self.n

  @property
  def hy(self) -> float:
    """Grid spacing in y."""
    return self.ay / self.m

  @property
  def cell_area(self) -> float:
    """Quadrature weight of one grid cell."""
    return self.hx * self.hy

  @property
  def shape(self) -> tuple[int, int]:
    """Shape ``(m, n)`` of the grid, in row-major (y, x) order."""
    return (self.m, self.n)

  @property
  def n_dof(self) -> int:
    """Number of grid points."""
    return self.n * self.m

  # -- coordinates ---------------------------------------------------------
  @property
  def x(self) -> Array:
    """The x axis, of length ``n``."""
    return -self.ax / 2 + self.hx * jnp.arange(self.n)

  @property
  def y(self) -> Array:
    """The y axis, of length ``m``."""
    return -self.ay / 2 + self.hy * jnp.arange(self.m)

  @property
  def X(self) -> Array:
    """x-coordinate of every grid point, of shape ``(m, n)``."""
    return jnp.broadcast_to(self.x[None, :], self.shape)

  @property
  def Y(self) -> Array:
    """y-coordinate of every grid point, of shape ``(m, n)``."""
    return jnp.broadcast_to(self.y[:, None], self.shape)

  @property
  def XY(self) -> tuple[Array, Array]:
    """``(X, Y)``, matching ``jnp.meshgrid(x, y)``."""
    return self.X, self.Y

  # -- helpers -------------------------------------------------------------
  def grid(self, u: Array) -> Array:
    """Reshape a flat field to the ``(m, n)`` grid."""
    return u.reshape(self.shape)

  def ravel(self, u: Array) -> Array:
    """Flatten a gridded field."""
    return u.reshape((-1,))

  def sample(self, fn) -> Array:
    """Sample ``fn(X, Y)`` on the grid and return it flattened.

    Replaces the ``fn(params.X, params.Y).reshape(-1)`` idiom.
    """
    return self.ravel(fn(self.X, self.Y))
