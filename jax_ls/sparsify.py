"""Ying's sparsifying preconditioner for the Lippmann-Schwinger operator.

  L. Ying, "Sparsifying preconditioner for the Lippmann-Schwinger equation",
  Multiscale Model. Simul. 13(2), 644-660 (2015); arXiv:1408.4495

The idea
--------
The volume potential ``G`` is dense, but it is the Green's function of a local
operator, so a small local stencil applied to it nearly annihilates its far
field.  For each grid point ``p`` pick the 3x3 stencil ``alpha_p`` with

  (Acal G)_{pj} = sum_k alpha_{p,k} G_{p + o_k, j}  ~  0    for j far from p,

computed as the smallest left singular vector of the rows of ``G`` on the
stencil, restricted to the far columns.  Keeping only the near part gives a
sparse ``C ~ Acal G`` with the same 9-point pattern as ``Acal``.

Mapping onto this code
----------------------
`jax_ls` solves the *density* form ``A sigma = f`` with ``A = -I + omega^2 D_nu
G``, ``D_nu = diag(nu)``.  Ying sparsifies the *field* form ``-I + omega^2 G
D_nu``, which here is ``A^T`` because the discrete ``G`` is complex-symmetric.
So

  Acal A^T = -Acal + omega^2 Acal G D_nu  ~  -Acal + omega^2 C D_nu  =: M,

a 9-point sparse matrix, and therefore

  A^{-1}  ~  P := Acal^T M^{-T}.

One sparse LU of ``M``, solved transposed, and a stencil application -- no
inverse of ``Acal`` is ever needed.  The adjoint-like operator ``-I + omega^2
D_nu G^H`` of `jax_ls.apply_lipp_schwin_adj` is ``conj(A(conj nu))``, so it is
preconditioned by ``conj(P_{conj nu}(conj r))``, which for a real contrast is
the *same* factorisation.

What depends on what
--------------------
* `SparsifyingStencils` depend only on the grid and the kernel, i.e. on the
  frequency: build them once per frequency, eagerly (`create` is not jittable;
  it calls the kernel ~81 times and does nine small SVDs).
* The factorisation of ``M`` depends on ``nu``.  It happens on the host, inside
  a ``jax.pure_callback``, the first time a contrast is seen, and is cached on
  the hash of that contrast: every solve inside one Gauss-Newton linearisation
  shares one factorisation.  ``refactor_rtol`` optionally reuses a cached
  factorisation for any nearby contrast -- GMRES stays exact, only slower.

Differentiation
---------------
``M^{-T}`` is applied through `jax.lax.custom_linear_solve`, with ``M^T``
written in `jax.numpy`, so a preconditioned solve differentiates and transposes
exactly like an unpreconditioned one; the host callback is only ever asked to
*solve*, never to be differentiated.

Measured, n = 127 and ``omega = 80`` (see `tests/test_sparsify.py` and the
notebook): a smooth contrast of 4 drops from 1333 GMRES iterations to 11, and a
discontinuous disc of contrast 4 from more than 1000 to 18.
"""

import collections
import hashlib
import itertools
import threading
from dataclasses import dataclass
from collections.abc import Callable
from functools import partial

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import jax
import jax.numpy as jnp

from flax import struct

from . import nd_solver
from .domain import Domain
from .kernels import GreenKernel
from .nd_solver import NDFactor, NDStructure

Array = jax.Array

# Stencil offsets (dy, dx), in a fixed order shared by `Acal` and `C`.  Using
# one list for both is what keeps their entries aligned; the Julia reference
# implementation builds the two from differently ordered lists on four edges
# and two corners, which is harmless only while ``nu`` vanishes there.
OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))

# Default number of factorisations kept on the host.
DEFAULT_CACHE_SIZE = 4

# How ``M`` is factored and solved.
#   "superlu"       -- SciPy's SuperLU on the host, through `jax.pure_callback`,
#                      cached on the contrast.  Best on CPU.
#   "block_inverse" -- `jax_ls.nd_solver`: nested dissection with inverted fronts,
#                      factored and solved in `jax.numpy`.  The solve is batched
#                      GEMMs, so it stays on the device; meant for accelerators.
#   "auto"          -- "block_inverse" when JAX's default backend is not the CPU.
BACKENDS = ("superlu", "block_inverse")

_STRUCTURES: dict[tuple[int, int, int], NDStructure] = {}


def _structure(m: int, n: int, leaf: int) -> NDStructure:
  """The symbolic analysis of a grid, shared across frequencies.

  Args:
    m: Number of rows.
    n: Number of columns.
    leaf: Leaf size of the dissection tree.

  Returns:
    The cached `NDStructure`, built on first use.
  """
  if (m, n, leaf) not in _STRUCTURES:
    _STRUCTURES[m, n, leaf] = NDStructure.create(m, n, leaf)
  return _STRUCTURES[m, n, leaf]


# Partial-pivoting threshold of the factorisation.  SuperLU pivots off the
# diagonal only when the diagonal entry is smaller than this fraction of the
# column maximum, so a small value keeps the nested-dissection order -- and its
# fill -- almost intact while still guarding against a vanishing pivot.
PIVOT_THRESHOLD = 0.01


def nested_dissection(m: int, n: int, leaf: int = 8) -> np.ndarray:
  """A geometric nested-dissection ordering of an ``m x n`` grid.

  Splits the longer side by a one-line separator, recursively, and numbers the
  separator after both halves.  With 9-point coupling a single line separates,
  so this is the ordering Ying's ``O(N^1.5)`` factorisation cost assumes.  The
  same tree drives `jax_ls.nd_solver`.

  Measured at n = 255 (and 511) against SuperLU's own orderings: factorisation
  0.5 s (6.8 s) against 1.5-5 s (11-16 s) for COLAMD and 59 s for
  MMD_AT_PLUS_A, with 40 % less fill than COLAMD.

  Args:
    m: Number of rows of the grid.
    n: Number of columns.
    leaf: Blocks with at most ``leaf**2`` points are not split further.

  Returns:
    A permutation of ``range(m * n)``: position ``i`` holds the old index.
  """
  return nd_solver.nested_dissection(m, n, leaf)


class _PermutedLU:
  """An LU of ``M[perm][:, perm]``, solving with ``M`` itself.

  ``M = Q^T M_p Q`` for the permutation matrix ``Q``, so ``op(M) x = b`` is
  ``op(M_p) (Q x) = Q b`` for ``op`` any of identity, transpose and conjugate
  transpose: the same two index shuffles serve all three.
  """

  def __init__(self, matrix: sps.csr_matrix, perm: np.ndarray):
    """Factor a matrix in a given symmetric order.

    Args:
      matrix: The sparse matrix ``M``.
      perm: Elimination order, as from `nested_dissection`.
    """
    self.perm = perm
    self.lu = spla.splu(
      matrix[perm][:, perm].tocsc(),
      permc_spec="NATURAL",
      diag_pivot_thresh=PIVOT_THRESHOLD,
      options=dict(SymmetricMode=True),
    )

  def solve(self, b: np.ndarray, trans: str = "N") -> np.ndarray:
    """Solve ``op(M) x = b``.

    Args:
      b: Right-hand sides, ``(N,)`` or ``(N, p)``.
      trans: ``"N"``, ``"T"`` or ``"H"`` for ``M``, ``M^T`` or ``M^H``.

    Returns:
      The solution, of the shape of `b`.
    """
    permuted = self.lu.solve(np.ascontiguousarray(b[self.perm]), trans=trans)
    x = np.empty_like(permuted)
    x[self.perm] = permuted
    return x


def _stencil_type(index: int, size: int) -> int:
  """Classify a grid index along one axis.

  Args:
    index: Position along the axis.
    size: Number of points along the axis.

  Returns:
    0 on the low boundary, 1 inside, 2 on the high boundary.
  """
  if index == 0:
    return 0
  if index == size - 1:
    return 2
  return 1


def _valid_offsets(domain: Domain, iy: int, ix: int) -> np.ndarray:
  """Which of `OFFSETS` stay on the grid around one point.

  Args:
    domain: The grid.
    iy: Row of the point.
    ix: Column of the point.

  Returns:
    A boolean array of length 9.
  """
  return np.array(
    [0 <= iy + dy < domain.m and 0 <= ix + dx < domain.n for dy, dx in OFFSETS]
  )


def _valid_mask(domain: Domain) -> np.ndarray:
  """Whether each offset stays on the grid, at every point.

  Args:
    domain: The grid.

  Returns:
    A boolean array of shape ``(9, m, n)``.
  """
  iy = np.arange(domain.m)[:, None]
  ix = np.arange(domain.n)[None, :]
  return np.stack(
    [
      (0 <= iy + dy)
      & (iy + dy < domain.m)
      & (0 <= ix + dx)
      & (ix + dx < domain.n)
      for dy, dx in OFFSETS
    ]
  )


# ---------------------------------------------------------------------------
# Stencil application in jax.numpy (so it differentiates and transposes).
# ---------------------------------------------------------------------------
def apply_stencil(domain: Domain, coefficients: Array, v: Array) -> Array:
  """``(S v)_p = sum_k coefficients[k, p] v[p + o_k]``.

  Args:
    domain: The grid.
    coefficients: Stencil weights, of shape ``(9, m, n)``, zero wherever the
      neighbour falls off the grid.
    v: A flat field.

  Returns:
    The flat field ``S v``.
  """
  m, n = domain.shape
  padded = jnp.pad(v.reshape((m, n)), 1)
  out = sum(
    coefficients[k] * padded[1 + dy : 1 + dy + m, 1 + dx : 1 + dx + n]
    for k, (dy, dx) in enumerate(OFFSETS)
  )
  return out.reshape((-1,))


def apply_stencil_transpose(
  domain: Domain, coefficients: Array, w: Array
) -> Array:
  """``S^T w``, the (non-conjugated) transpose of `apply_stencil`.

  Args:
    domain: The grid.
    coefficients: Stencil weights, of shape ``(9, m, n)``, zero wherever the
      neighbour falls off the grid.
    w: A flat field.

  Returns:
    The flat field ``S^T w``.
  """
  m, n = domain.shape
  w = w.reshape((m, n))
  out = 0.0
  for k, (dy, dx) in enumerate(OFFSETS):
    padded = jnp.pad(coefficients[k] * w, 1)
    out = out + padded[1 - dy : 1 - dy + m, 1 - dx : 1 - dx + n]
  return out.reshape((-1,))


# ---------------------------------------------------------------------------
# Host side: assembled matrices and the factorisation cache.
# ---------------------------------------------------------------------------
@dataclass
class _HostEntry:
  """What the host keeps for one set of stencils.

  Args:
    stencil_matrix: ``Acal`` as a CSR matrix.
    local_matrix: ``C`` as a CSR matrix.
    refactor_rtol: Relative distance within which a cached factorisation is
      reused for a new contrast.
    perm: Nested-dissection elimination order of the grid.
  """

  stencil_matrix: sps.csr_matrix
  local_matrix: sps.csr_matrix
  refactor_rtol: float
  perm: np.ndarray


@dataclass
class _Factor:
  """One cached factorisation.

  Args:
    key: Handle of the stencils it was built from.
    nu: The contrast it was built for.
    omega: The frequency it was built for.
    lu: The factorisation.
  """

  key: int
  nu: np.ndarray
  omega: float
  lu: _PermutedLU


class _Registry:
  """Assembled stencil matrices and an LRU cache of factorisations.

  Kept on the host because a SuperLU object is not something JAX can carry; the
  device side refers to it through the integer `SparsifyingStencils.key`.
  """

  def __init__(self, cache_size: int = DEFAULT_CACHE_SIZE):
    """An empty registry.

    Args:
      cache_size: How many factorisations to keep.
    """
    self._lock = threading.Lock()
    self._counter = itertools.count()
    self.entries: dict[int, _HostEntry] = {}
    self.factors: collections.OrderedDict = collections.OrderedDict()
    self.cache_size = cache_size
    self.stats = collections.Counter()

  def register(self, entry: _HostEntry) -> int:
    """Store the matrices of one set of stencils.

    Args:
      entry: What to keep.

    Returns:
      The handle under which it is kept.
    """
    with self._lock:
      key = next(self._counter)
      self.entries[key] = entry
      return key

  def factor(self, key: int, omega: float, nu: np.ndarray) -> _PermutedLU:
    """The factorisation of ``M`` for one set of stencils and one contrast.

    Args:
      key: Handle of the stencils.
      omega: Angular frequency.
      nu: Flat contrast.

    Returns:
      The factorisation, from the cache when possible.
    """
    digest = hashlib.blake2b(
      np.ascontiguousarray(nu).tobytes() + np.float64(omega).tobytes(),
      digest_size=16,
    ).hexdigest()
    with self._lock:
      cache_key = (key, nu.dtype.str, digest)
      if cache_key in self.factors:
        self.factors.move_to_end(cache_key)
        self.stats["hits"] += 1
        return self.factors[cache_key].lu

      entry = self.entries[key]
      if entry.refactor_rtol > 0:
        for cached in reversed(self.factors.values()):
          if cached.key != key or cached.omega != omega:
            continue
          scale = max(np.linalg.norm(cached.nu), np.finfo(float).tiny)
          if np.linalg.norm(nu - cached.nu) <= entry.refactor_rtol * scale:
            self.stats["reuses"] += 1
            return cached.lu

      matrix = -entry.stencil_matrix + omega**2 * (
        entry.local_matrix @ sps.diags(nu)
      )
      lu = _PermutedLU(matrix.tocsr(), entry.perm)
      self.stats["factorizations"] += 1
      self.factors[cache_key] = _Factor(key, np.array(nu), omega, lu)
      while len(self.factors) > self.cache_size:
        self.factors.popitem(last=False)
      return lu


_REGISTRY = _Registry()


def factorization_stats() -> dict[str, int]:
  """How the host cache has been used.

  Returns:
    Counts of host ``"factorizations"``, cache ``"hits"`` and tolerance-based
    ``"reuses"`` for the ``"superlu"`` backend, and of
    ``"device_factorizations"`` for the ``"block_inverse"`` backend.
  """
  return dict(_REGISTRY.stats)


def _record_device_factorization() -> None:
  """Count one ``"block_inverse"`` factorisation, at run time.

  A `jax.debug.callback`, so it counts executions of compiled code -- what
  caching saves -- rather than traces.
  """

  def bump() -> None:
    """Increment the counter."""
    with _REGISTRY._lock:
      _REGISTRY.stats["device_factorizations"] += 1

  jax.debug.callback(bump)


def clear_factorizations() -> None:
  """Drop every cached factorisation and reset the counters."""
  with _REGISTRY._lock:
    _REGISTRY.factors.clear()
    _REGISTRY.stats.clear()


def set_cache_size(size: int) -> None:
  """Set how many factorisations to keep; each costs ``O(N log N)`` memory.

  Args:
    size: The new capacity; the oldest entries beyond it are dropped.

  Raises:
    ValueError: If `size` is smaller than one.
  """
  if size < 1:
    raise ValueError(
      f"the cache must hold at least one factorisation, got {size}"
    )
  with _REGISTRY._lock:
    _REGISTRY.cache_size = size
    while len(_REGISTRY.factors) > size:
      _REGISTRY.factors.popitem(last=False)


def _host_solve(
  trans: str,
  key: np.ndarray,
  omega: np.ndarray,
  nu: np.ndarray,
  rhs: np.ndarray,
) -> np.ndarray:
  """Solve ``M x = rhs`` (or transposed) on the host.

  Called through `jax.pure_callback` with ``vmap_method="expand_dims"``, so
  every argument may carry leading batch dimensions, of size one when it is not
  batched.  The common case -- one contrast, many sources -- becomes a single
  SuperLU solve with a block right-hand side.

  Args:
    trans: ``"N"``, ``"T"`` or ``"H"``.
    key: Handle of the stencils.
    omega: Angular frequency.
    nu: Contrast, ``(..., N)``.
    rhs: Right-hand sides, ``(..., N)``.

  Returns:
    The solutions, of the broadcast shape of `nu` and `rhs`.
  """
  key = int(np.ravel(key)[0])
  omega = float(np.ravel(omega)[0])
  n_dof = rhs.shape[-1]
  nu = np.asarray(nu)
  rhs = np.asarray(rhs)
  out_dtype = rhs.dtype
  batch = np.broadcast_shapes(nu.shape[:-1], rhs.shape[:-1])

  nu_rows = nu.reshape((-1, n_dof))
  if nu_rows.shape[0] == 1:
    lu = _REGISTRY.factor(key, omega, nu_rows[0])
    block = np.broadcast_to(rhs, batch + (n_dof,)).reshape((-1, n_dof))
    solution = lu.solve(np.asarray(block.T, dtype=np.complex128), trans=trans)
    return solution.T.reshape(batch + (n_dof,)).astype(out_dtype)

  nu_rows = np.broadcast_to(nu, batch + (n_dof,)).reshape((-1, n_dof))
  rhs_rows = np.broadcast_to(rhs, batch + (n_dof,)).reshape((-1, n_dof))
  solution = np.stack(
    [
      _REGISTRY.factor(key, omega, row).solve(
        np.asarray(b, dtype=np.complex128), trans=trans
      )
      for row, b in zip(nu_rows, rhs_rows)
    ]
  )
  return solution.reshape(batch + (n_dof,)).astype(out_dtype)


# ---------------------------------------------------------------------------
# The stencils.
# ---------------------------------------------------------------------------
@struct.dataclass
class SparsifyingStencils:
  """The ``nu``-independent part of the preconditioner, for one frequency.

  Args:
    domain: The grid.
    alpha: Weights of ``Acal``, of shape ``(9, m, n)``, zero off the grid.
    local: Weights of ``C ~ Acal G`` on the same pattern, of shape
      ``(9, m, n)``.
    quality: ``s_min / s_max`` of the far-field block for each of the nine
      stencil types, ordered (bottom-left, bottom, bottom-right, left, interior,
      right, top-left, top, top-right) in grid order.  The fraction of the far
      field the stencil fails to annihilate; small is good.
    key: Handle of the assembled matrices on the host.  A leaf rather than a
      static field so that changing frequency does not force a recompile.
    structure: The nested-dissection analysis of the grid, for the
      ``"block_inverse"`` backend; None otherwise.
    backend: How ``M`` is factored; see `BACKENDS`.
  """

  domain: Domain
  alpha: Array
  local: Array
  quality: Array
  key: Array
  structure: NDStructure | None = None
  backend: str = struct.field(pytree_node=False, default="superlu")

  @classmethod
  def create(
    cls,
    kernel: GreenKernel,
    refactor_rtol: float = 0.0,
    backend: str = "auto",
    leaf: int = nd_solver.DEFAULT_LEAF,
  ) -> "SparsifyingStencils":
    """Build the stencils for a kernel.

    `G` is a convolution, so a stencil depends only on which neighbours exist:
    there are nine types (interior, four edges, four corners), each computed
    once at a representative point and replicated.

    Args:
      kernel: The truncated Green's kernel of one frequency.
      refactor_rtol: Reuse a cached factorisation for any contrast within this
        relative distance of the one it was built for.  0 refactors for every
        new contrast.  ``"superlu"`` backend only.  See `jax_ls.sparsify`.
      backend: ``"superlu"``, ``"block_inverse"`` or ``"auto"``; see
        `BACKENDS`.
      leaf: Leaf size of the dissection tree, ``"block_inverse"`` only.

    Returns:
      The stencils, with their matrices registered on the host.

    Raises:
      ValueError: If the grid has fewer than three points along an axis, or
        the backend is unknown.
    """
    if backend == "auto":
      backend = "superlu" if jax.default_backend() == "cpu" else "block_inverse"
    if backend not in BACKENDS:
      raise ValueError(
        f"backend must be one of {BACKENDS} or 'auto', got {backend!r}"
      )
    domain = kernel.domain
    m, n = domain.shape
    if m < 3 or n < 3:
      raise ValueError(
        f"the sparsifying stencils need at least 3x3 points, got {m}x{n}"
      )
    n_dof = m * n
    green = jax.jit(kernel.apply)
    dtype = kernel.ghat.dtype

    alpha_types, local_types, quality = {}, {}, []
    for ty, iy in enumerate((0, m // 2, m - 1)):
      for tx, ix in enumerate((0, n // 2, n - 1)):
        valid = _valid_offsets(domain, iy, ix)
        points = np.array(
          [
            (iy + dy) * n + (ix + dx)
            for (dy, dx), ok in zip(OFFSETS, valid)
            if ok
          ]
        )
        # Columns of G at the stencil points are its rows there, G being
        # symmetric; each column is one application of the kernel.
        rows = np.stack(
          [
            np.asarray(green(jnp.zeros(n_dof, dtype).at[p].set(1.0)))
            for p in points
          ]
        )
        far = np.setdiff1d(np.arange(n_dof), points)
        u, s, _ = np.linalg.svd(rows[:, far], full_matrices=False)
        weights = np.conj(u[:, -1])
        # A singular vector is defined only up to a complex phase, and the SVD
        # does not pick the same one every time.  The preconditioner does not
        # care (a row phase cancels in Acal^T M^{-T}), but reproducibility does:
        # fix the phase so the centre weight is real and positive.
        centre = weights[list(points).index(iy * n + ix)]
        weights = weights * (abs(centre) / centre)
        alpha_types[ty, tx] = (valid, weights)
        local_types[ty, tx] = weights @ rows[:, points]
        quality.append(s[-1] / s[0])

    alpha = np.zeros((len(OFFSETS), m, n), dtype=np.complex128)
    local = np.zeros_like(alpha)
    type_y = np.array([_stencil_type(i, m) for i in range(m)])
    type_x = np.array([_stencil_type(i, n) for i in range(n)])
    for (ty, tx), (valid, weights) in alpha_types.items():
      where = (type_y[:, None] == ty) & (type_x[None, :] == tx)
      for k in np.flatnonzero(valid):
        j = np.count_nonzero(valid[:k])
        alpha[k][where] = weights[j]
        local[k][where] = local_types[ty, tx][j]

    key = _REGISTRY.register(
      _HostEntry(
        stencil_matrix=_assemble(domain, alpha),
        local_matrix=_assemble(domain, local),
        refactor_rtol=float(refactor_rtol),
        perm=nested_dissection(m, n),
      )
    )
    return cls(
      domain=domain,
      alpha=jnp.asarray(alpha, dtype=dtype),
      local=jnp.asarray(local, dtype=dtype),
      quality=jnp.asarray(quality),
      key=jnp.asarray(key, dtype=jnp.int32),
      structure=_structure(m, n, leaf) if backend == "block_inverse" else None,
      backend=backend,
    )

  # -- the sparse matrices, for inspection and tests ------------------------
  def matrices(self) -> tuple[sps.csr_matrix, sps.csr_matrix]:
    """The assembled stencil matrices.

    Returns:
      ``(Acal, C)`` as SciPy CSR matrices.
    """
    entry = _REGISTRY.entries[int(self.key)]
    return entry.stencil_matrix, entry.local_matrix

  def apply_alpha(self, v: Array) -> Array:
    """Apply ``Acal``.

    Args:
      v: A flat field.

    Returns:
      ``Acal v``.
    """
    return apply_stencil(self.domain, self.alpha, v)

  def apply_local(self, v: Array) -> Array:
    """Apply ``C``.

    Args:
      v: A flat field.

    Returns:
      ``C v``.
    """
    return apply_stencil(self.domain, self.local, v)


def _assemble(domain: Domain, coefficients: np.ndarray) -> sps.csr_matrix:
  """Assemble a 9-point stencil field as a sparse matrix.

  Args:
    domain: The grid.
    coefficients: Stencil weights, ``(9, m, n)``.

  Returns:
    The ``N x N`` CSR matrix with ``M[p, p + o_k] = coefficients[k, p]``.
  """
  m, n = domain.shape
  valid = _valid_mask(domain)
  points = np.arange(m * n).reshape((m, n))
  rows, cols, vals = [], [], []
  for k, (dy, dx) in enumerate(OFFSETS):
    mask = valid[k]
    rows.append(points[mask])
    cols.append(points[mask] + dy * n + dx)
    vals.append(coefficients[k][mask])
  return sps.csr_matrix(
    (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
    shape=(m * n, m * n),
  )


# ---------------------------------------------------------------------------
# The preconditioner.
# ---------------------------------------------------------------------------
def matrix_coefficients(
  stencils: SparsifyingStencils, omega: float | Array, nu: Array
) -> Array:
  """The 9-point field of ``M = -Acal + omega^2 C D_nu``.

  ``M[p, p + o_k] = -alpha[k, p] + omega^2 local[k, p] nu[p + o_k]``.

  Args:
    stencils: The sparsifying stencils of this frequency.
    omega: Angular frequency.
    nu: Flat contrast.

  Returns:
    An array of shape ``(9, m, n)``.
  """
  m, n = stencils.domain.shape
  padded = jnp.pad(jnp.reshape(nu, (m, n)), 1)
  shifted = jnp.stack(
    [padded[1 + dy : 1 + dy + m, 1 + dx : 1 + dx + n] for dy, dx in OFFSETS]
  )
  return -stencils.alpha + omega**2 * stencils.local * shifted


def factor_matrix(
  stencils: SparsifyingStencils, omega: float | Array, nu: Array
) -> NDFactor:
  """The ``"block_inverse"`` factorisation of ``M`` at a contrast.

  Exposed for inspection: `NDFactor.growth` and `NDFactor.pivot_ratio` report
  how well the per-front pivoting did.  Works whatever the backend of
  `stencils`.

  Args:
    stencils: The sparsifying stencils of this frequency.
    omega: Angular frequency.
    nu: Flat contrast.

  Returns:
    The factorisation of ``M``.
  """
  structure = stencils.structure
  if structure is None:
    structure = _structure(*stencils.domain.shape, nd_solver.DEFAULT_LEAF)
  return nd_solver.factorize(
    structure, matrix_coefficients(stencils, omega, nu)
  )


@struct.dataclass
class PreconditionerFactors:
  """Factorisations of ``M`` kept for reuse across solves at one contrast.

  Every solve built on a ``"block_inverse"`` preconditioner factors ``M`` from
  the contrast it is given.  When several solves share a contrast -- the
  forward and tangent solves of a JVP, the forward and adjoint solves of a
  gradient, every ``J`` and ``J^H`` application of a Gauss-Newton step -- this
  object lets them share one factorisation instead.  It must have been built
  for the very contrast it is used with; nothing checks.

  Args:
    forward: Factorisation of ``M(nu)``, for the forward operator.
    adjoint: Factorisation of ``M(conj nu)``, for the adjoint-like operator.
      The same object as `forward` when the contrast is real.
  """

  forward: NDFactor
  adjoint: NDFactor

  def swapped(self) -> "PreconditionerFactors":
    """The factors seen from the adjoint operator.

    Returns:
      The same factors with `forward` and `adjoint` exchanged.
    """
    return PreconditionerFactors(forward=self.adjoint, adjoint=self.forward)


def factorize_preconditioner(
  stencils: SparsifyingStencils, omega: float | Array, nu: Array
) -> PreconditionerFactors | None:
  """Factor ``M`` once, for every solve at this contrast.

  Args:
    stencils: The sparsifying stencils of this frequency.
    omega: Angular frequency.
    nu: Flat contrast.

  Returns:
    The factors for the ``"block_inverse"`` backend.  None for ``"superlu"``,
    whose host cache already reuses a factorisation across calls.
  """
  if stencils.backend != "block_inverse":
    return None
  omega_c, nu_c = jax.lax.stop_gradient((omega, nu))
  forward = factor_matrix(stencils, omega_c, nu_c)
  _record_device_factorization()
  if not jnp.iscomplexobj(nu_c):
    return PreconditionerFactors(forward=forward, adjoint=forward)
  adjoint = factor_matrix(stencils, omega_c, jnp.conj(nu_c))
  _record_device_factorization()
  return PreconditionerFactors(forward=forward, adjoint=adjoint)


def _inverse_mt(
  stencils: SparsifyingStencils,
  omega: float | Array,
  nu: Array,
  factor: NDFactor | None = None,
) -> Callable[[Array], Array]:
  """``r -> M^{-T} r`` with ``M = -Acal + omega^2 C D_nu``.

  The factorisation, when it is done in JAX, happens here -- once -- and the
  returned callable closes over it; GMRES then applies it as a loop invariant.

  Args:
    stencils: The sparsifying stencils of this frequency.
    omega: Angular frequency.
    nu: Flat contrast.
    factor: A ``"block_inverse"`` factorisation of ``M`` at `nu` to reuse; when
      None, one is computed here.

  Returns:
    A callable applying ``M^{-T}``, differentiable and transposable through
    `jax.lax.custom_linear_solve`.
  """
  domain = stencils.domain
  # The solver only ever receives constants: derivatives with respect to the
  # contrast come from `matvec`, which `custom_linear_solve` differentiates.
  key, omega_c, nu_c = jax.lax.stop_gradient((stencils.key, omega, nu))

  def matvec(v: Array) -> Array:
    """Apply ``M^T = -Acal^T + omega^2 D_nu C^T`` in `jax.numpy`.

    Args:
      v: A flat vector.

    Returns:
      ``M^T v``.
    """
    return -apply_stencil_transpose(
      domain, stencils.alpha, v
    ) + omega**2 * nu * apply_stencil_transpose(domain, stencils.local, v)

  if stencils.backend == "block_inverse":
    if factor is None:
      factor = nd_solver.factorize(
        stencils.structure, matrix_coefficients(stencils, omega_c, nu_c)
      )
      _record_device_factorization()

    def device(mode: str) -> Callable:
      """A `custom_linear_solve` solver applying ``op(M)^{-1}`` on device.

      Args:
        mode: ``"N"`` or ``"T"``.

      Returns:
        A function of ``(matvec, b)`` that ignores `matvec`.
      """
      return lambda _, b: nd_solver.solve(factor, b, mode)

    solve, transpose_solve = device("T"), device("N")
  else:
    omega_c = jnp.asarray(omega_c, dtype=float)

    def host(trans: str) -> Callable:
      """A `custom_linear_solve` solver applying ``op(M)^{-1}`` on the host.

      Args:
        trans: ``"N"`` or ``"T"``.

      Returns:
        A function of ``(matvec, b)`` that ignores `matvec`.
      """

      def solve(_: Callable, b: Array) -> Array:
        """Solve ``op(M) x = b`` through `jax.pure_callback`.

        Args:
          _: The mat-vec `custom_linear_solve` passes; unused.
          b: Right-hand side.

        Returns:
          The solution.
        """
        return jax.pure_callback(
          partial(_host_solve, trans),
          jax.ShapeDtypeStruct(b.shape, b.dtype),
          key,
          omega_c,
          nu_c,
          b,
          vmap_method="expand_dims",
        )

      return solve

    solve, transpose_solve = host("T"), host("N")

  return lambda r: jax.lax.custom_linear_solve(
    matvec, r, solve=solve, transpose_solve=transpose_solve
  )


def preconditioner(
  stencils: SparsifyingStencils,
  omega: float | Array,
  nu: Array,
  adjoint: bool = False,
  factors: PreconditionerFactors | None = None,
) -> Callable[[Array], Array]:
  """The preconditioner as a callable, ready for `jax_ls.solvers.solve`.

  ``P = Acal^T M^{-T}`` approximates the inverse of ``-I + omega^2 D_nu G``.
  With `adjoint`, the inverse of ``-I + omega^2 D_nu G^H = conj(A(conj nu))``
  is approximated by ``conj(P_{conj nu}(conj r))`` -- for a real contrast, the
  same factorisation as the forward one.

  Any factorisation in JAX is done once, here, not on every application.

  Args:
    stencils: The sparsifying stencils of this frequency.
    omega: Angular frequency.
    nu: Flat contrast.
    adjoint: Precondition the adjoint-like operator instead.
    factors: Factorisations from `factorize_preconditioner` at this same
      contrast, to reuse instead of factoring again.  Ignored by the
      ``"superlu"`` backend.

  Returns:
    A callable applying the preconditioner to a flat vector.
  """
  if adjoint:
    forward = preconditioner(
      stencils,
      omega,
      jnp.conj(nu),
      factors=None if factors is None else factors.swapped(),
    )
    return lambda r: jnp.conj(forward(jnp.conj(r)))

  inverse = _inverse_mt(
    stencils, omega, nu, None if factors is None else factors.forward
  )

  def apply(r: Array) -> Array:
    """Apply ``P = Acal^T M^{-T}``.

    Args:
      r: A flat vector.

    Returns:
      ``P r``.
    """
    r = jnp.asarray(r)
    r = r.astype(jnp.result_type(r.dtype, stencils.alpha.dtype))
    return apply_stencil_transpose(stencils.domain, stencils.alpha, inverse(r))

  return apply


def apply_preconditioner(
  stencils: SparsifyingStencils, omega: float | Array, nu: Array, r: Array
) -> Array:
  """``P r = Acal^T M^{-T} r``, an approximate inverse of ``-I + omega^2 D_nu G``.

  Builds the preconditioner for this one application; to apply it repeatedly,
  call `preconditioner` once instead.

  Args:
    stencils: The sparsifying stencils of this frequency.
    omega: Angular frequency.
    nu: Flat contrast.
    r: Flat vector.

  Returns:
    ``P r``.
  """
  return preconditioner(stencils, omega, nu)(r)


def apply_preconditioner_adjoint(
  stencils: SparsifyingStencils, omega: float | Array, nu: Array, r: Array
) -> Array:
  """Approximate inverse of ``-I + omega^2 D_nu G^H``; see `preconditioner`.

  Args:
    stencils: The sparsifying stencils of this frequency.
    omega: Angular frequency.
    nu: Flat contrast.
    r: Flat vector.

  Returns:
    The preconditioned vector.
  """
  return preconditioner(stencils, omega, nu, adjoint=True)(r)
