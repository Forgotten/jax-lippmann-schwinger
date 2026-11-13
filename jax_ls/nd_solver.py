"""A nested-dissection direct solver for 9-point matrices, with inverted fronts.

The solve this module produces uses no triangular solves: only gathers,
batched dense matrix products and scatter-adds.  On a GPU that is the
difference between a chain of small dependent steps and a handful of batched
GEMMs, and it is what lets a solve live inside a jitted, `vmap`-ed JAX program
with no host round trip.  It is the factorisation behind
``SparsifyingStencils.create(..., backend="block_inverse")``.

The factorisation
-----------------
Order the grid by geometric nested dissection (`NDStructure`).  A tree node has

* a *separator* ``S`` -- a line of cells splitting its box in two, or the whole
  box for a leaf -- and
* a *boundary* ``B`` -- the cells adjacent to its box, all of which lie in
  ancestor separators.

Eliminating children before parents gives, per node, a dense *front*

  F = [F_SS  F_SB]   (original entries with an end in S, plus the children's
      [F_BS  F_BB]    Schur complements, extend-added)

and ``M = L D U`` with

  D^{-1} = blockdiag(F_SS^{-1}),
  L      = I + sum_t E_B lower_t E_S^T,     lower = F_BS F_SS^{-1},
  U      = I + sum_t E_S upper_t E_B^T,     upper = F_SS^{-1} F_SB,

the Schur complement ``F_BB - F_BS upper`` passing up to the parent.  Those
three dense blocks per node (`LevelFactor.inverse`, `.lower`, `.upper`), plus
the integer index sets of ``S`` and ``B``, are the *whole* factor: ``L`` and
``U`` are never formed, and their sparsity pattern is implicit in the index
sets.  Then

  M x = b:    up the tree     b_B -= lower b_S;    z_S = inverse b_S
              down the tree   x_S  = z_S - upper x_B
  M^T x = b:  the same with (upper^T, inverse^T, lower^T);
  M^H x = b:  with conjugates as well.

Pivoting
--------
Each ``F_SS`` is factored with partial pivoting (`jax.lax.linalg.lu`), and the
three stored blocks are formed by triangular solves against that LU -- the
backward-stable way, rather than by multiplying with an inverse.  The
permutations are folded into the stored blocks (``F_SS^{-1} = U^{-1} L^{-1}
P``), so the solve carries no permutation arrays.  Pivots never cross from
``S`` into ``B``: delayed pivoting would make block sizes data dependent and
break the static batched layout.  Measured on sparsifying matrices up to
contrast 32, this keeps element growth below 4, against up to 780 without
pivoting; `NDFactor` reports the worst growth and pivot ratio so that a failure
is visible rather than silent.

Layout
------
Nodes are grouped by depth.  Within a depth they are independent, so each level
is padded to its largest ``s = |S|`` and ``b = |B|`` and stacked: a level is
five arrays, ``separator_idx (k, s)``, ``boundary_idx (k, b)``, ``inverse (k,
s, s)``, ``lower (k, b, s)`` and ``upper (k, s, b)``.  Padded indices point at a
dummy slot ``N`` appended to every vector, so padded reads see zero and padded
writes are discarded.  Padding costs 2-10 % of the storage; the storage itself
grows like ``N log N`` -- about 30 MiB at n = 127 and 140 MiB at n = 255 in
complex128 with ``leaf = 8``.

Jitting
-------
`NDStructure.create` is host-side numpy, run once per grid.  `factorize` and
`solve` are `jax.jit`-compiled and take the structure and the factor as
ordinary pytree arguments; they compile once per grid and leaf size.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from flax import struct

Array = jax.Array

# Stencil offsets (dy, dx), in the order shared with `jax_ls.sparsify`.
OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))

# Default size below which a box is not split: boxes of at most ``leaf**2``
# points become leaves.  8 gives 7x7 leaves.
DEFAULT_LEAF = 8

MODES = ("N", "T", "H")


# ---------------------------------------------------------------------------
# The dissection tree.
# ---------------------------------------------------------------------------
@dataclass
class TreeNode:
  """One node of the dissection tree, on the host; see `dissection_tree`.

  Args:
    separator: Grid indices of the node's separator, or of its whole box for a
      leaf.
    boundary: Grid indices adjacent to the node's box, all in ancestor
      separators.
    depth: Distance from the root.
    children: The two halves the separator splits the box into.
  """

  separator: np.ndarray
  boundary: np.ndarray
  depth: int
  children: list["TreeNode"] = field(default_factory=list)

  def nodes(self) -> Iterator["TreeNode"]:
    """Every node of this subtree, children before parents.

    Returns:
      An iterator in elimination order.
    """
    return _postorder(self)


def _build_tree(m: int, n: int, leaf: int) -> TreeNode:
  """Build the dissection tree of an ``m x n`` grid.

  Splits the longer side of each box by a one-line separator, recursively, until
  a box has at most ``leaf**2`` points.

  Args:
    m: Number of rows of the grid.
    n: Number of columns.
    leaf: Boxes of at most ``leaf**2`` points are not split.

  Returns:
    The root of the tree.
  """
  ids = np.arange(m * n).reshape((m, n))

  def adjacent(r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    """Grid indices adjacent to the box ``[r0, r1) x [c0, c1)``.

    Args:
      r0: First row of the box.
      r1: One past its last row.
      c0: First column.
      c1: One past its last column.

    Returns:
      The indices of the ring of cells around the box that lie on the grid.
    """
    rows = np.arange(max(r0 - 1, 0), min(r1 + 1, m))
    cols = np.arange(max(c0 - 1, 0), min(c1 + 1, n))
    ring = ids[np.ix_(rows, cols)].ravel()
    return np.setdiff1d(ring, ids[r0:r1, c0:c1].ravel())

  def split(r0: int, r1: int, c0: int, c1: int, depth: int) -> TreeNode:
    """The subtree of the box ``[r0, r1) x [c0, c1)``.

    Args:
      r0: First row of the box.
      r1: One past its last row.
      c0: First column.
      c1: One past its last column.
      depth: Depth of the box's node.

    Returns:
      The node of the box, with its children built.
    """
    n_rows, n_cols = r1 - r0, c1 - c0
    boundary = adjacent(r0, r1, c0, c1)
    if n_rows * n_cols <= leaf * leaf or (n_rows < 3 and n_cols < 3):
      return TreeNode(ids[r0:r1, c0:c1].ravel(), boundary, depth)
    if n_cols >= n_rows:
      mid = c0 + n_cols // 2
      node = TreeNode(ids[r0:r1, mid].ravel(), boundary, depth)
      halves = [(r0, r1, c0, mid), (r0, r1, mid + 1, c1)]
    else:
      mid = r0 + n_rows // 2
      node = TreeNode(ids[mid, c0:c1].ravel(), boundary, depth)
      halves = [(r0, mid, c0, c1), (mid + 1, r1, c0, c1)]
    children = [split(*half, depth + 1) for half in halves]
    node.children = [child for child in children if child.separator.size]
    return node

  return split(0, m, 0, n, 0)


def _postorder(node: TreeNode) -> Iterator[TreeNode]:
  """Visit a tree children first: the elimination order.

  Args:
    node: Root of the (sub)tree.

  Returns:
    An iterator over the nodes, every child before its parent.
  """
  for child in node.children:
    yield from _postorder(child)
  yield node


def dissection_tree(m: int, n: int, leaf: int = DEFAULT_LEAF) -> TreeNode:
  """The nested-dissection tree of an ``m x n`` grid, for inspection.

  The same tree `NDStructure.create` and `nested_dissection` are built from.

  Args:
    m: Number of rows of the grid.
    n: Number of columns.
    leaf: Boxes of at most ``leaf**2`` points are not split.

  Returns:
    The root node; `TreeNode.nodes` walks the tree in elimination order.
  """
  return _build_tree(m, n, leaf)


def nested_dissection(m: int, n: int, leaf: int = DEFAULT_LEAF) -> np.ndarray:
  """The elimination order of the dissection tree, as a permutation.

  Args:
    m: Number of rows of the grid.
    n: Number of columns.
    leaf: Boxes of at most ``leaf**2`` points are not split.

  Returns:
    A permutation of ``range(m * n)``: position ``i`` holds the old index of the
    ``i``-th unknown eliminated.
  """
  root = _build_tree(m, n, leaf)
  return np.concatenate([node.separator for node in _postorder(root)])


# ---------------------------------------------------------------------------
# The symbolic structure: everything that depends on the grid only.
# ---------------------------------------------------------------------------
@struct.dataclass
class LevelStructure:
  """Index maps of one depth of the tree, padded to uniform shapes.

  Args:
    separator_idx: Separator indices, ``(k, s)``, padded with the dummy index
      ``N``.
    boundary_idx: Boundary indices, ``(k, b)``, padded with ``N``.
    entry_source: For each original matrix entry assembled at this level, its
      flat index into the ``(9, N)`` coefficient field.
    entry_target: Where that entry lands in the flattened ``(k, f, f)`` fronts,
      ``f = s + b``.
    child_target: Where each entry of the children's ``(k_c, b_c, b_c)`` Schur
      complements lands in this level's flattened fronts; ``k f f``, one past
      the end, for padding.  Empty at the deepest level.
    pad_diagonal: Flat indices of the padded diagonal of ``F_SS``, set to one so
      that the batched factorisation is defined.
    separator_valid: Boolean ``(k, s)``, false on padding.
  """

  separator_idx: Array
  boundary_idx: Array
  entry_source: Array
  entry_target: Array
  child_target: Array
  pad_diagonal: Array
  separator_valid: Array

  @property
  def shape(self) -> tuple[int, int, int]:
    """The padded sizes of this level.

    Returns:
      ``(k, s, b)``: the number of fronts, and the padded sizes of their
      separators and boundaries.  Static under `jax.jit`, being array shapes.
    """
    k, s = self.separator_idx.shape
    return k, s, self.boundary_idx.shape[1]


@struct.dataclass
class NDStructure:
  """The symbolic analysis of a grid: tree, levels and assembly maps.

  Depends only on the grid shape and the leaf size, so it is built once and
  shared by every factorisation on that grid -- across contrasts and across
  frequencies.  Building it (`create`) is host-side numpy and not jittable; the
  result is a pytree of integer arrays that `factorize` and `solve` take as an
  ordinary argument under `jax.jit`.

  Args:
    levels: One `LevelStructure` per depth, deepest first.
    m: Rows of the grid.
    n: Columns of the grid.
    leaf: Leaf size the tree was built with.
  """

  levels: tuple[LevelStructure, ...]
  m: int = struct.field(pytree_node=False)
  n: int = struct.field(pytree_node=False)
  leaf: int = struct.field(pytree_node=False, default=DEFAULT_LEAF)

  @property
  def n_dof(self) -> int:
    """Number of unknowns.

    Returns:
      ``m * n``.
    """
    return self.m * self.n

  @property
  def stored_entries(self) -> int:
    """Size of a factorisation built on this structure.

    Returns:
      The number of complex entries of the stored blocks over all levels,
      padding included.
    """
    return sum(
      k * (s * s + 2 * s * b) for k, s, b in (lv.shape for lv in self.levels)
    )

  @classmethod
  def create(cls, m: int, n: int, leaf: int = DEFAULT_LEAF) -> "NDStructure":
    """Analyse an ``m x n`` grid with 9-point coupling.

    Args:
      m: Number of rows.
      n: Number of columns.
      leaf: Boxes of at most ``leaf**2`` points are not split.

    Returns:
      The structure, with every index map as a device array.

    Raises:
      ValueError: If `leaf` is smaller than 2.
    """
    if leaf < 2:
      raise ValueError(f"the leaf size must be at least 2, got {leaf}")
    n_dof = m * n
    nodes = list(_postorder(_build_tree(m, n, leaf)))

    by_depth: dict[int, list[TreeNode]] = {}
    for node in nodes:
      by_depth.setdefault(node.depth, []).append(node)
    depths = sorted(by_depth, reverse=True)  # deepest first
    padded_sizes = {
      d: (
        max(len(x.separator) for x in by_depth[d]),
        max(max((len(x.boundary) for x in by_depth[d]), default=0), 1),
      )
      for d in depths
    }

    # Node numbers are postorder positions.  Which node holds each unknown in
    # its separator, and at which position.
    number = {id(node): t for t, node in enumerate(nodes)}
    index_in_level = {}
    for d in depths:
      for i, node in enumerate(by_depth[d]):
        index_in_level[id(node)] = i
    level_index = np.array([index_in_level[id(node)] for node in nodes])
    depth_of = np.array([node.depth for node in nodes])
    node_of = np.empty(n_dof, dtype=np.int64)
    slot_in_separator = np.empty(n_dof, dtype=np.int64)
    for t, node in enumerate(nodes):
      node_of[node.separator] = t
      slot_in_separator[node.separator] = np.arange(len(node.separator))

    # Position of a boundary unknown inside a node's boundary, via a sorted key
    # (node, unknown) -> position.
    keys = np.concatenate(
      [t * (n_dof + 1) + node.boundary for t, node in enumerate(nodes)]
    )
    positions = np.concatenate(
      [np.arange(len(node.boundary)) for node in nodes]
    )
    order = np.argsort(keys)
    keys, positions = keys[order], positions[order]

    def front_slot(
      owners: np.ndarray, unknowns: np.ndarray, s_pad: int
    ) -> np.ndarray:
      """Where unknowns sit in the fronts of the nodes that hold them.

      Args:
        owners: Node numbers, one per unknown.
        unknowns: Grid indices, each in the separator or the boundary of the
          matching node.
        s_pad: The padded separator size of the nodes' level.

      Returns:
        The front slot of each unknown: its position in the separator, or
        `s_pad` plus its position in the boundary.

      Raises:
        AssertionError: If an unknown is in neither, which would be a bug in
          the tree.
      """
      in_separator = node_of[unknowns] == owners
      slots = slot_in_separator[unknowns].copy()
      wanted = owners[~in_separator] * (n_dof + 1) + unknowns[~in_separator]
      where = np.searchsorted(keys, wanted)
      if np.any(keys[np.minimum(where, len(keys) - 1)] != wanted):
        raise AssertionError("an unknown is in neither part of its front")
      slots[~in_separator] = s_pad + positions[where]
      return slots

    # Original entries M[i, i + o_k], assembled in the front of whichever end
    # is eliminated first -- the deeper node, since one end's node is always an
    # ancestor of the other's.
    iy, ix = np.divmod(np.arange(n_dof), n)
    rows, offsets, cols = [], [], []
    for k, (dy, dx) in enumerate(OFFSETS):
      on_grid = (0 <= iy + dy) & (iy + dy < m) & (0 <= ix + dx) & (ix + dx < n)
      i = np.flatnonzero(on_grid)
      rows.append(i)
      offsets.append(np.full(i.size, k))
      cols.append(i + dy * n + dx)
    rows, offsets, cols = map(np.concatenate, (rows, offsets, cols))
    owner = np.where(
      depth_of[node_of[rows]] >= depth_of[node_of[cols]],
      node_of[rows],
      node_of[cols],
    )
    owner_depth = depth_of[owner]

    levels = []
    for d in depths:
      level_nodes = by_depth[d]
      k_level = len(level_nodes)
      s_level, b_level = padded_sizes[d]
      f_level = s_level + b_level

      separator_idx = np.full((k_level, s_level), n_dof)
      boundary_idx = np.full((k_level, b_level), n_dof)
      separator_valid = np.zeros((k_level, s_level), dtype=bool)
      pad_diagonal = []
      for i, node in enumerate(level_nodes):
        n_sep = len(node.separator)
        separator_idx[i, :n_sep] = node.separator
        boundary_idx[i, : len(node.boundary)] = node.boundary
        separator_valid[i, :n_sep] = True
        for slot in range(n_sep, s_level):
          pad_diagonal.append((i * f_level + slot) * f_level + slot)

      here = owner_depth == d
      local = level_index[owner[here]]
      row_slot = front_slot(owner[here], rows[here], s_level)
      col_slot = front_slot(owner[here], cols[here], s_level)
      entry_source = offsets[here] * n_dof + rows[here]
      entry_target = (local * f_level + row_slot) * f_level + col_slot

      # Extend-add maps from the level below.
      if d + 1 in by_depth:
        child_nodes = by_depth[d + 1]
        b_child = padded_sizes[d + 1][1]
        child_target = np.full(
          (len(child_nodes), b_child, b_child), k_level * f_level * f_level
        )
        parent_of = {
          id(child): node for node in level_nodes for child in node.children
        }
        for ci, child in enumerate(child_nodes):
          parent = parent_of[id(child)]
          n_bnd = len(child.boundary)
          slots = front_slot(
            np.full(n_bnd, number[id(parent)]), child.boundary, s_level
          )
          base = index_in_level[id(parent)] * f_level
          child_target[ci, :n_bnd, :n_bnd] = (
            base + slots[:, None]
          ) * f_level + slots[None, :]
        child_target = child_target.ravel()
      else:
        child_target = np.zeros(0, dtype=np.int64)

      levels.append(
        LevelStructure(
          separator_idx=jnp.asarray(separator_idx, dtype=jnp.int32),
          boundary_idx=jnp.asarray(boundary_idx, dtype=jnp.int32),
          entry_source=jnp.asarray(entry_source, dtype=jnp.int32),
          entry_target=jnp.asarray(entry_target, dtype=jnp.int32),
          child_target=jnp.asarray(child_target, dtype=jnp.int32),
          pad_diagonal=jnp.asarray(
            np.array(pad_diagonal, dtype=np.int64), dtype=jnp.int32
          ),
          separator_valid=jnp.asarray(separator_valid),
        )
      )
    return cls(levels=tuple(levels), m=m, n=n, leaf=leaf)


# ---------------------------------------------------------------------------
# The numeric factorisation.
# ---------------------------------------------------------------------------
@struct.dataclass
class LevelFactor:
  """The stored blocks of one depth of the tree; see the module docstring.

  Args:
    separator_idx: Separator indices, ``(k, s)``, padded with the dummy index
      ``N``.
    boundary_idx: Boundary indices, ``(k, b)``, padded with ``N``.
    inverse: Inverted diagonal blocks ``F_SS^{-1}``, ``(k, s, s)``, with the
      pivoting folded in and zero padded rows.
    lower: Blocks of ``L``, ``F_BS F_SS^{-1}``, ``(k, b, s)``.
    upper: Blocks of ``U``, ``F_SS^{-1} F_SB``, ``(k, s, b)``.
  """

  separator_idx: Array
  boundary_idx: Array
  inverse: Array
  lower: Array
  upper: Array


@struct.dataclass
class NDFactor:
  """A factorisation ``M = L D U``, stored as inverted fronts.

  Args:
    levels: One `LevelFactor` per depth, deepest first.
    growth: Worst element growth ``max|U| / max|F_SS|`` over all fronts.
    pivot_ratio: Worst ``min|u_kk| / max|u_kk|`` over all fronts.
  """

  levels: tuple[LevelFactor, ...]
  growth: Array
  pivot_ratio: Array

  def solve(self, rhs: Array, mode: str = "N") -> Array:
    """Solve with the factored matrix; see the module-level `solve`.

    Args:
      rhs: Right-hand side, ``(N,)`` or ``(N, p)``.
      mode: ``"N"``, ``"T"`` or ``"H"`` for ``M``, ``M^T`` or ``M^H``.

    Returns:
      ``op(M)^{-1} rhs``, of the shape of `rhs`.
    """
    return solve(self, rhs, mode)


def _batched_take(x: Array, perm: Array) -> Array:
  """Permute the rows of a stack of matrices.

  Args:
    x: Matrices, ``(k, s, c)``.
    perm: One row permutation per matrix, ``(k, s)``.

  Returns:
    ``x[i, perm[i], :]`` for every ``i``.
  """
  return jnp.take_along_axis(x, perm[..., None], axis=1)


def _factor_level(
  fronts: Array, s: int, separator_valid: Array
) -> tuple[Array, Array, Array, Array, Array, Array]:
  """Eliminate the separator block of every front of one level.

  ``F_SS`` is factored with partial pivoting; the three stored blocks come from
  triangular solves against that LU.

  Args:
    fronts: Assembled fronts, ``(k, f, f)`` with ``f = s + b``.
    s: Padded separator size.
    separator_valid: Boolean ``(k, s)``, false on padding.

  Returns:
    ``(inverse, lower, upper, update, growth, pivot_ratio)``: the stored blocks;
    the Schur complements ``F_BB - F_BS upper``, ``(k, b, b)``, passed to the
    parent level; and the worst element growth and pivot ratio of the level.
  """
  f_ss, f_sb = fronts[:, :s, :s], fronts[:, :s, s:]
  f_bs, f_bb = fronts[:, s:, :s], fronts[:, s:, s:]

  lu, _, perm = jax.lax.linalg.lu(f_ss)
  inverse_perm = jnp.argsort(perm, axis=-1)

  def solve_n(x: Array) -> Array:
    """``F_SS^{-1} x``, using ``F_SS[perm] = L U``.

    Args:
      x: Right-hand sides, ``(k, s, c)``.

    Returns:
      The solutions, ``(k, s, c)``.
    """
    y = _batched_take(x, perm)
    y = jax.lax.linalg.triangular_solve(
      lu, y, left_side=True, lower=True, unit_diagonal=True
    )
    return jax.lax.linalg.triangular_solve(lu, y, left_side=True, lower=False)

  def solve_t(x: Array) -> Array:
    """``F_SS^{-T} x``.

    Args:
      x: Right-hand sides, ``(k, s, c)``.

    Returns:
      The solutions, ``(k, s, c)``.
    """
    y = jax.lax.linalg.triangular_solve(
      lu, x, left_side=True, lower=False, transpose_a=True
    )
    y = jax.lax.linalg.triangular_solve(
      lu, y, left_side=True, lower=True, transpose_a=True, unit_diagonal=True
    )
    return _batched_take(y, inverse_perm)

  identity = jnp.broadcast_to(jnp.eye(s, dtype=fronts.dtype), f_ss.shape)
  # Zero the padded rows of the inverse so that padded slots stay inert.
  inverse = solve_n(identity) * separator_valid[:, :, None]
  upper = solve_n(f_sb)
  lower = jnp.swapaxes(solve_t(jnp.swapaxes(f_bs, 1, 2)), 1, 2)
  update = f_bb - f_bs @ upper

  diagonal = jnp.abs(jnp.diagonal(lu, axis1=1, axis2=2))
  largest = jnp.max(jnp.where(separator_valid, diagonal, 0.0), axis=1)
  smallest = jnp.min(jnp.where(separator_valid, diagonal, jnp.inf), axis=1)
  growth = jnp.max(jnp.abs(jnp.triu(lu)), axis=(1, 2)) / jnp.maximum(
    jnp.max(jnp.abs(f_ss) * separator_valid[:, :, None], axis=(1, 2)),
    jnp.finfo(diagonal.dtype).tiny,
  )
  return (
    inverse,
    lower,
    upper,
    update,
    jnp.max(growth),
    jnp.min(smallest / largest),
  )


@jax.jit
def factorize(structure: NDStructure, coefficients: Array) -> NDFactor:
  """Factor the 9-point matrix ``M[i, i + o_k] = coefficients[k, i]``.

  Jitted, and pure `jax.numpy`, so it runs on whatever device JAX does.  It
  compiles once per grid and leaf size: the structure's index arrays are
  ordinary traced inputs whose *shapes* fix the program.  The cost is dominated
  by the batched LU and products of the top few levels.

  Args:
    structure: The symbolic analysis of the grid.
    coefficients: The stencil weights, of shape ``(9, m, n)`` or ``(9, N)``;
      entries pointing off the grid are ignored.

  Returns:
    The factorisation.
  """
  values = jnp.reshape(coefficients, (-1,))
  dtype = jnp.result_type(values.dtype, jnp.complex64)
  values = values.astype(dtype)

  levels, update = [], None
  growth = jnp.asarray(1.0, dtype=jnp.finfo(dtype).dtype)
  pivot_ratio = jnp.asarray(1.0, dtype=growth.dtype)
  for lv in structure.levels:
    k, s, b = lv.shape
    f = s + b
    flat = jnp.zeros(k * f * f + 1, dtype=dtype)
    flat = flat.at[lv.entry_target].add(values[lv.entry_source])
    if update is not None:
      flat = flat.at[lv.child_target].add(update.reshape(-1))
    flat = flat.at[lv.pad_diagonal].set(1.0)
    fronts = flat[:-1].reshape((k, f, f))

    inverse, lower, upper, update, level_growth, level_ratio = _factor_level(
      fronts, s, lv.separator_valid
    )
    growth = jnp.maximum(growth, level_growth)
    pivot_ratio = jnp.minimum(pivot_ratio, level_ratio)
    levels.append(
      LevelFactor(
        separator_idx=lv.separator_idx,
        boundary_idx=lv.boundary_idx,
        inverse=inverse,
        lower=lower,
        upper=upper,
      )
    )
  return NDFactor(levels=tuple(levels), growth=growth, pivot_ratio=pivot_ratio)


# ---------------------------------------------------------------------------
# The solve.
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("mode",))
def solve(factor: NDFactor, rhs: Array, mode: str = "N") -> Array:
  """Solve ``op(M) x = rhs`` using only gathers, batched GEMMs and scatters.

  Args:
    factor: The factorisation of ``M``.
    rhs: One right-hand side of shape ``(N,)`` or several of shape ``(N, p)``.
      Under `jax.vmap` a single right-hand side is batched into GEMMs too.
    mode: ``"N"`` for ``M``, ``"T"`` for ``M^T``, ``"H"`` for ``M^H``.

  Returns:
    The solution, of the shape of `rhs`.

  Raises:
    ValueError: If `mode` is not one of ``"N"``, ``"T"`` and ``"H"``.
  """
  if mode not in MODES:
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
  if mode == "H":
    return jnp.conj(solve(factor, jnp.conj(rhs), "T"))

  single = rhs.ndim == 1
  x = rhs[:, None] if single else rhs
  dtype = jnp.result_type(x.dtype, factor.levels[0].inverse.dtype)
  x = jnp.concatenate([x.astype(dtype), jnp.zeros((1, x.shape[1]), dtype)])
  dummy = x.shape[0] - 1
  transposed = mode == "T"

  def transpose(a: Array) -> Array:
    """Transpose every matrix of a stack.

    Args:
      a: Matrices, ``(k, r, c)``.

    Returns:
      ``(k, c, r)``.
    """
    return jnp.swapaxes(a, 1, 2)

  pending = []
  for lv in factor.levels:  # up the tree, deepest level first
    y = x[lv.separator_idx]  # (k, s, p)
    up = transpose(lv.upper) if transposed else lv.lower
    inverse = transpose(lv.inverse) if transposed else lv.inverse
    x = x.at[lv.boundary_idx].add(-(up @ y))
    x = x.at[dummy].set(0.0)
    pending.append(inverse @ y)
  for lv, z in zip(reversed(factor.levels), reversed(pending)):  # down
    down = transpose(lv.lower) if transposed else lv.upper
    x = x.at[lv.separator_idx].set(z - down @ x[lv.boundary_idx])
    x = x.at[dummy].set(0.0)

  out = x[:dummy]
  return out[:, 0] if single else out
