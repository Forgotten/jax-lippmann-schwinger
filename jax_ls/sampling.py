"""Where the medium is probed from, and where the field is measured.

Two roles that the near-field map used to conflate into a single ``G_sample``
matrix:

* an `IncidentField` says *what illuminates the medium* -- one column of
  ``(n_dof, n_sources)`` per source;
* a `SamplingManifold` says *where the scattered field is measured* -- it
  produces a sampling kernel of ``(n_dof, n_receivers)``.

They are separate objects because a wide-band sweep wants different counts for
each: the resolution analysis of Borges, Gillman and Greengard asks for
``N_src ~ 2*omega*a/pi`` incident directions but ``N_recv ~ 4*omega*a/pi``
receivers.

Sign convention
---------------
Both matrices store **minus** the outgoing fundamental solution, matching the
original ``G_sample = -green_function(R)``.  The two signs cancel in the
near-field product, so the convention is invisible as long as the data comes
from this same forward map.  It is *not* invisible against externally measured
data: the map returns minus the physical scattered field.  Preserved here
deliberately rather than fixed, so that every existing result is unchanged.
"""

from typing import Protocol

import jax
import jax.numpy as jnp

from flax import struct

from .domain import Domain
from .kernels import green_function

Array = jax.Array


@struct.dataclass
class SamplingManifold:
  """A set of points in the plane, used for sources or for receivers.

  Attributes:
    points: The point coordinates, of shape ``(n_points, 2)``.
  """

  points: Array

  # -- constructors --------------------------------------------------------
  @classmethod
  def circle(
    cls, radius: float, n_points: int, center: tuple[float, float] = (0.0, 0.0)
  ) -> "SamplingManifold":
    """`n_points` equispaced points on a circle.

    The parametrisation starts at angle ``pi`` and wraps once, matching the
    original `init_params_near_field`.
    """
    d_theta = 2.0 * jnp.pi / n_points
    theta = jnp.linspace(jnp.pi, 3 * jnp.pi - d_theta, n_points)
    return cls.from_angles(radius, theta, center)

  @classmethod
  def arc(
    cls,
    radius: float,
    n_points: int,
    theta_start: float,
    theta_end: float,
    center: tuple[float, float] = (0.0, 0.0),
  ) -> "SamplingManifold":
    """`n_points` equispaced points on an arc -- a limited aperture."""
    theta = jnp.linspace(theta_start, theta_end, n_points)
    return cls.from_angles(radius, theta, center)

  @classmethod
  def from_angles(
    cls, radius: float, theta: Array, center: tuple[float, float] = (0.0, 0.0)
  ) -> "SamplingManifold":
    """Points at the given angles on a circle of the given radius."""
    points = radius * jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)
    return cls(points=points + jnp.asarray(center).reshape((1, 2)))

  @classmethod
  def line(
    cls, start: tuple[float, float], end: tuple[float, float], n_points: int
  ) -> "SamplingManifold":
    """`n_points` equispaced points on a segment -- a transmission geometry."""
    t = jnp.linspace(0.0, 1.0, n_points).reshape((-1, 1))
    p0 = jnp.asarray(start).reshape((1, 2))
    p1 = jnp.asarray(end).reshape((1, 2))
    return cls(points=p0 + t * (p1 - p0))

  @classmethod
  def from_points(cls, points: Array) -> "SamplingManifold":
    """An arbitrary set of points, of shape ``(n_points, 2)``."""
    points = jnp.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
      raise ValueError(
        f"points must have shape (n_points, 2), got {points.shape}"
      )
    return cls(points=points)

  # -- use ---
  @property
  def n_points(self) -> int:
    """Number of points on the manifold."""
    return self.points.shape[0]

  def __len__(self) -> int:
    return self.n_points

  def distances_to(self, domain: Domain) -> Array:
    """Distance from every grid point to every manifold point.

    Returns:
      An array of shape ``(n_dof, n_points)``.
    """
    return jnp.sqrt(
      jnp.square(domain.X.reshape((-1, 1)) - self.points[:, 0].reshape((1, -1)))
      + jnp.square(
        domain.Y.reshape((-1, 1)) - self.points[:, 1].reshape((1, -1))
      )
    )

  def sampling_kernel(self, domain: Domain, omega: float) -> Array:
    """The kernel that samples a density on the grid at these points.

    Returns:
      ``-G`` between every grid point and every manifold point, of shape
      ``(n_dof, n_points)``.  See the module docstring on the sign.
    """
    # `green_function` goes through `scipy.special.hankel1`, so the result comes
    # back as a host numpy array.  Commit it to the device here: left as numpy it
    # is a pytree leaf that every jitted call would copy host -> device afresh,
    # which is free on CPU but not on an accelerator -- this matrix is several
    # megabytes and is used by every solve.
    return jnp.asarray(-green_function(self.distances_to(domain), omega))


class IncidentField(Protocol):
  """What illuminates the medium.

  Named for the code's own vocabulary: ``u_i`` in every equation of the module
  docstrings, and `_incident_rhs` in `near_field`.  Not "Illumination", which is
  optical and in the seismic literature usually means aperture coverage, and not
  "Source", which collides with the induced source ``sigma = omega^2 nu u``.
  """

  def evaluate(self, domain: Domain, omega: float) -> Array:
    """The incident field of every source, of shape ``(n_dof, n_sources)``."""
    ...

  @property
  def n_sources(self) -> int:
    """Number of illuminations."""
    ...


@struct.dataclass
class PointSources(IncidentField):
  """Point sources placed on a manifold.

  Attributes:
    manifold: Where the sources sit.
  """

  manifold: SamplingManifold

  @property
  def n_sources(self) -> int:
    return self.manifold.n_points

  def evaluate(self, domain: Domain, omega: float) -> Array:
    return self.manifold.sampling_kernel(domain, omega)


@struct.dataclass
class PlaneWaves(IncidentField):
  """Plane waves travelling along given directions.

  Unlike `PointSources` this has no positions at all, which is why the manifold
  lives inside the implementation rather than being hoisted into the problem.

  Attributes:
    directions: Unit propagation directions, of shape ``(n_sources, 2)``.
  """

  directions: Array

  @classmethod
  def from_angles(cls, theta: Array) -> "PlaneWaves":
    """Plane waves travelling at the given angles."""
    return cls(directions=jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1))

  @property
  def n_sources(self) -> int:
    return self.directions.shape[0]

  def evaluate(self, domain: Domain, omega: float) -> Array:
    phase = domain.X.reshape((-1, 1)) * self.directions[:, 0].reshape(
      (1, -1)
    ) + domain.Y.reshape((-1, 1)) * self.directions[:, 1].reshape((1, -1))
    # Negated to share the sign convention of `PointSources`; see the module
    # docstring.  An overall sign on the incident field cancels in the
    # near-field product.
    return -jnp.exp(1j * omega * phase)
