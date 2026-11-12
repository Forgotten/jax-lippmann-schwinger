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
from functools import partial

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import jax
import jax.numpy as jnp

from flax import struct

from .domain import Domain
from .kernels import GreenKernel

Array = jax.Array

# Stencil offsets (dy, dx), in a fixed order shared by `Acal` and `C`.  Using
# one list for both is what keeps their entries aligned; the Julia reference
# implementation builds the two from differently ordered lists on four edges
# and two corners, which is harmless only while ``nu`` vanishes there.
OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))

# Default number of factorisations kept on the host.
DEFAULT_CACHE_SIZE = 4

# Partial-pivoting threshold of the factorisation.  SuperLU pivots off the
# diagonal only when the diagonal entry is smaller than this fraction of the
# column maximum, so a small value keeps the nested-dissection order -- and its
# fill -- almost intact while still guarding against a vanishing pivot.
PIVOT_THRESHOLD = 0.01


def nested_dissection(m: int, n: int, leaf: int = 8) -> np.ndarray:
  """A geometric nested-dissection ordering of an ``m x n`` grid.

  Splits the longer side by a one-line separator, recursively, and numbers the
  separator after both halves.  With 9-point coupling a single line separates,
  so this is the ordering Ying's ``O(N^1.5)`` factorisation cost assumes.

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
  order = []

  def split(block):
    rows, cols = block.shape
    if rows * cols <= leaf * leaf:
      order.append(block.ravel())
    elif cols >= rows:
      mid = cols // 2
      split(block[:, :mid])
      split(block[:, mid + 1 :])
      order.append(block[:, mid])
    else:
      mid = rows // 2
      split(block[:mid])
      split(block[mid + 1 :])
      order.append(block[mid])

  split(np.arange(m * n).reshape((m, n)))
  return np.concatenate(order)


class _PermutedLU:
  """An LU of ``M[perm][:, perm]``, solving with ``M`` itself.

  ``M = Q^T M_p Q`` for the permutation matrix ``Q``, so ``op(M) x = b`` is
  ``op(M_p) (Q x) = Q b`` for ``op`` any of identity, transpose and conjugate
  transpose: the same two index shuffles serve all three.
  """

  def __init__(self, M: sps.csr_matrix, perm: np.ndarray):
    self.perm = perm
    self.lu = spla.splu(
      M[perm][:, perm].tocsc(),
      permc_spec="NATURAL",
      diag_pivot_thresh=PIVOT_THRESHOLD,
      options=dict(SymmetricMode=True),
    )

  def solve(self, b: np.ndarray, trans: str = "N") -> np.ndarray:
    permuted = self.lu.solve(np.ascontiguousarray(b[self.perm]), trans=trans)
    x = np.empty_like(permuted)
    x[self.perm] = permuted
    return x


def _stencil_type(index: int, size: int) -> int:
  """0 on the low boundary, 1 inside, 2 on the high boundary."""
  if index == 0:
    return 0
  if index == size - 1:
    return 2
  return 1


def _valid_offsets(domain: Domain, iy: int, ix: int) -> np.ndarray:
  """Which of `OFFSETS` stay on the grid around the point (iy, ix)."""
  return np.array(
    [0 <= iy + dy < domain.m and 0 <= ix + dx < domain.n for dy, dx in OFFSETS]
  )


def _valid_mask(domain: Domain) -> np.ndarray:
  """``(9, m, n)``: whether each offset stays on the grid at each point."""
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
  """``S^T w``, the (non-conjugated) transpose of `apply_stencil`."""
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
  """What the host keeps for one set of stencils."""

  A: sps.csr_matrix
  C: sps.csr_matrix
  refactor_rtol: float
  perm: np.ndarray


@dataclass
class _Factor:
  """One cached factorisation."""

  key: int
  nu: np.ndarray
  omega: float
  lu: object


class _Registry:
  """Assembled stencil matrices and an LRU cache of factorisations.

  Kept on the host because a SuperLU object is not something JAX can carry; the
  device side refers to it through the integer `SparsifyingStencils.key`.
  """

  def __init__(self, cache_size: int = DEFAULT_CACHE_SIZE):
    self._lock = threading.Lock()
    self._counter = itertools.count()
    self.entries: dict[int, _HostEntry] = {}
    self.factors: collections.OrderedDict = collections.OrderedDict()
    self.cache_size = cache_size
    self.stats = collections.Counter()

  def register(self, entry: _HostEntry) -> int:
    with self._lock:
      key = next(self._counter)
      self.entries[key] = entry
      return key

  def factor(self, key: int, omega: float, nu: np.ndarray):
    """The LU for (stencils, omega, nu), from the cache when possible."""
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

      M = (-entry.A + omega**2 * (entry.C @ sps.diags(nu))).tocsr()
      lu = _PermutedLU(M, entry.perm)
      self.stats["factorizations"] += 1
      self.factors[cache_key] = _Factor(key, np.array(nu), omega, lu)
      while len(self.factors) > self.cache_size:
        self.factors.popitem(last=False)
      return lu


_REGISTRY = _Registry()


def factorization_stats() -> dict[str, int]:
  """Counts of factorisations, cache hits and tolerance-based reuses."""
  return dict(_REGISTRY.stats)


def clear_factorizations() -> None:
  """Drop every cached factorisation and reset the counters."""
  with _REGISTRY._lock:
    _REGISTRY.factors.clear()
    _REGISTRY.stats.clear()


def set_cache_size(size: int) -> None:
  """How many factorisations to keep; each costs ``O(N log N)`` memory."""
  if size < 1:
    raise ValueError(
      f"the cache must hold at least one factorisation, got {size}"
    )
  with _REGISTRY._lock:
    _REGISTRY.cache_size = size
    while len(_REGISTRY.factors) > size:
      _REGISTRY.factors.popitem(last=False)


def _host_solve(trans: str, key, omega, nu, rhs):
  """Solve ``M x = rhs`` (or transposed) on the host.

  Called through `jax.pure_callback` with ``vmap_method="expand_dims"``, so
  every argument may carry leading batch dimensions, of size one when it is not
  batched.  The common case -- one contrast, many sources -- becomes a single
  SuperLU solve with a block right-hand side.
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

  Attributes:
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
  """

  domain: Domain
  alpha: Array
  local: Array
  quality: Array
  key: Array

  @classmethod
  def create(
    cls, kernel: GreenKernel, refactor_rtol: float = 0.0
  ) -> "SparsifyingStencils":
    """Build the stencils for a kernel.

    `G` is a convolution, so a stencil depends only on which neighbours exist:
    there are nine types (interior, four edges, four corners), each computed
    once at a representative point and replicated.

    Args:
      kernel: The truncated Green's kernel of one frequency.
      refactor_rtol: Reuse a cached factorisation for any contrast within this
        relative distance of the one it was built for.  0 refactors for every
        new contrast.  See `jax_ls.sparsify`.

    Returns:
      The stencils, with their matrices registered on the host.

    Raises:
      ValueError: If the grid has fewer than three points along an axis.
    """
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
        A=_assemble(domain, alpha),
        C=_assemble(domain, local),
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
    )

  # -- the sparse matrices, for inspection and tests ------------------------
  def matrices(self) -> tuple[sps.csr_matrix, sps.csr_matrix]:
    """``(Acal, C)`` as SciPy CSR matrices."""
    entry = _REGISTRY.entries[int(self.key)]
    return entry.A, entry.C

  def apply_A(self, v: Array) -> Array:
    """``Acal v``."""
    return apply_stencil(self.domain, self.alpha, v)

  def apply_C(self, v: Array) -> Array:
    """``C v``."""
    return apply_stencil(self.domain, self.local, v)


def _assemble(domain: Domain, coefficients: np.ndarray) -> sps.csr_matrix:
  """The CSR matrix of a 9-point stencil field."""
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
def _apply_inverse_MT(
  stencils: SparsifyingStencils, omega, nu: Array, r: Array
) -> Array:
  """``M^{-T} r`` with ``M = -Acal + omega^2 C D_nu``."""
  domain = stencils.domain
  # The callback only ever receives constants: derivatives with respect to the
  # contrast come from `matvec`, which `custom_linear_solve` differentiates.
  key, omega_c, nu_c = jax.lax.stop_gradient((stencils.key, omega, nu))
  omega_c = jnp.asarray(omega_c, dtype=float)

  def matvec(v):  # M^T v = -Acal^T v + omega^2 D_nu C^T v
    return -apply_stencil_transpose(
      domain, stencils.alpha, v
    ) + omega**2 * nu * apply_stencil_transpose(domain, stencils.local, v)

  def host(trans):
    def solve(_, b):
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

  return jax.lax.custom_linear_solve(
    matvec, r, solve=host("T"), transpose_solve=host("N")
  )


def apply_preconditioner(
  stencils: SparsifyingStencils, omega, nu: Array, r: Array
) -> Array:
  """``P r = Acal^T M^{-T} r``, an approximate inverse of ``-I + omega^2 D_nu G``.

  Args:
    stencils: The sparsifying stencils of this frequency.
    omega: Angular frequency.
    nu: Flat contrast.
    r: Flat vector.

  Returns:
    ``P r``.
  """
  r = jnp.asarray(r)
  dtype = jnp.result_type(r.dtype, stencils.alpha.dtype)
  r = r.astype(dtype)
  return apply_stencil_transpose(
    stencils.domain, stencils.alpha, _apply_inverse_MT(stencils, omega, nu, r)
  )


def apply_preconditioner_adjoint(
  stencils: SparsifyingStencils, omega, nu: Array, r: Array
) -> Array:
  """Approximate inverse of ``-I + omega^2 D_nu G^H``.

  That operator is ``conj(A(conj nu))``, so this is ``conj(P_{conj nu}(conj
  r))`` -- for a real contrast, the same factorisation as the forward one.
  """
  return jnp.conj(
    apply_preconditioner(stencils, omega, jnp.conj(nu), jnp.conj(r))
  )


def preconditioner(
  stencils: SparsifyingStencils, omega, nu: Array, adjoint: bool = False
):
  """The preconditioner as a callable, ready for `jax_ls.solvers.solve`."""
  apply = apply_preconditioner_adjoint if adjoint else apply_preconditioner
  return partial(apply, stencils, omega, nu)
