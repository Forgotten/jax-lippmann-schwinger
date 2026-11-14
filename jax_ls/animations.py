"""Animations for the notebooks: nested dissection, and inversions as they run.

Not imported by ``jax_ls`` itself, since it needs matplotlib (and Pillow to
write GIFs), which the solver does not: ``pip install "jax-ls[viz]"``, then

  from jax_ls import animations

Every animation is written as a GIF.  A notebook then shows it from a markdown
cell (``![...](images/name.gif)``), which plays in Jupyter, VS Code and on
GitHub alike -- unlike a JavaScript player, which GitHub does not run.  On
Colab, where relative image links do not resolve, `save_gif` also displays the
animation inline.

Three animations:

* `animate_dissection` -- a grid cut by separators, depth by depth, beside its
  dissection tree;
* `animate_elimination` -- the grid, the tree and the stored blocks of
  `jax_ls.nd_solver` through the factorisation and both passes of a solve;
* `animate_reconstruction` -- recorded inversions (`jax_ls.trajectory`) side by
  side, with their model error and misfit drawn as they decay.

Colour carries one job at a time: a single blue ramp for depth, which is
ordered; orange, blue and gray for the states of elimination; a diverging map
with a neutral midpoint for the contrast, which has a sign; and fixed series
colours, in a fixed order, for the runs being compared.
"""

import os
import sys
from collections.abc import Callable, Sequence

import numpy as np

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from .domain import Domain
from .inversion import Trajectory
from .nd_solver import TreeNode, dissection_tree

Array = jax.Array

SURFACE, INK, MUTED, EDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#d6d5d0"
PENDING, DONE, CURRENT, BOUNDARY = "#efeeeb", "#a8a79f", "#eb6834", "#2a78d6"
# Depth is ordered: one hue, dark at the root to light at the leaves.
DEPTH_RAMP = ["#0d366b", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6"]
# Runs being compared, in this fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
# The contrast has a sign: two poles and a neutral midpoint.
CONTRAST_MAP = LinearSegmentedColormap.from_list(
  "contrast", ["#b5462f", "#e8a690", "#f0efec", "#86b6ef", "#104281"]
)


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------
def rgb(color: str) -> np.ndarray:
  """A hex colour as an RGB triple.

  Args:
    color: ``"#rrggbb"``.

  Returns:
    The colour, with components in [0, 1].
  """
  color = color.lstrip("#")
  return np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0


def ease(t: float) -> float:
  """Smoothstep: slow in, slow out.

  Args:
    t: Progress, clipped to [0, 1].

  Returns:
    The eased progress.
  """
  t = np.clip(t, 0.0, 1.0)
  return t * t * (3 - 2 * t)


def mix(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
  """Blend colour arrays.

  Args:
    a: Colours at ``t = 0``.
    b: Colours at ``t = 1``.
    t: Blend factor.

  Returns:
    ``(1 - t) a + t b``.
  """
  return (1 - t) * a + t * b


def fade(before: np.ndarray, after: np.ndarray, t: float) -> np.ndarray:
  """Change colours through the neutral pending colour, never through a mix.

  Blending two categorical hues directly passes through a third colour that
  means nothing -- orange into blue goes via purple -- so a changing cell first
  fades out to the neutral colour, then in to its new one.

  Args:
    before: Colours at the start, ``(..., 3)``.
    after: Colours at the end, of the same shape.
    t: Progress in [0, 1].

  Returns:
    The colours at `t`; entries that do not change stay put.
  """
  t = ease(t)
  neutral = rgb(PENDING)
  out = np.where(
    t < 0.5,
    mix(before, neutral, np.clip(2 * t, 0, 1)),
    mix(neutral, after, np.clip(2 * t - 1, 0, 1)),
  )
  same = np.all(np.isclose(before, after), axis=-1, keepdims=True)
  return np.where(same, after, out)


def header(fig, title: str, subtitle: str) -> None:
  """Write the title and the one-line explanation of the current frame.

  Args:
    fig: The figure.
    title: Bold first line.
    subtitle: Second line.
  """
  fig.texts.clear()
  fig.text(0.02, 0.965, title, fontsize=14, color=INK, weight="bold", va="top")
  fig.text(0.02, 0.905, subtitle, fontsize=10.5, color=MUTED, va="top")


def progress(fig, fraction: float) -> None:
  """Draw a thin progress bar along the bottom of the figure.

  Args:
    fig: The figure.
    fraction: How much of the animation has played.
  """
  if not hasattr(fig, "_progress"):
    fig._progress = fig.add_axes((0.02, 0.015, 0.96, 0.02))
  ax = fig._progress
  ax.clear()
  ax.axis("off")
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.add_patch(plt.Rectangle((0, 0.35), 1, 0.3, color=PENDING))
  ax.add_patch(plt.Rectangle((0, 0.35), fraction, 0.3, color=MUTED))


def save_gif(
  fig, draw: Callable[[int], None], n_frames: int, path: str, fps: int, dpi: int
) -> str:
  """Render an animation to a GIF, and show it inline on Colab.

  Args:
    fig: The figure `draw` updates.
    draw: Draws frame ``i`` into `fig`.
    n_frames: Number of frames.
    path: Where to write the GIF; parent directories are created.
    fps: Frames per second.
    dpi: Resolution.

  Returns:
    `path`.
  """
  directory = os.path.dirname(path)
  if directory:
    os.makedirs(directory, exist_ok=True)
  movie = animation.FuncAnimation(
    fig, draw, frames=n_frames, interval=1000 / fps
  )
  movie.save(
    path,
    writer=animation.PillowWriter(fps=fps),
    dpi=dpi,
    savefig_kwargs={"facecolor": SURFACE},
  )
  plt.close(fig)
  if "google.colab" in sys.modules:  # relative image links fail there
    from IPython.display import Image, display

    display(Image(filename=path))
  return path


# ---------------------------------------------------------------------------
# Nested dissection.
# ---------------------------------------------------------------------------
class DissectionTree:
  """The dissection tree of a grid, with a layout for drawing it.

  Args:
    m: Rows of the grid.
    n: Columns of the grid.
    leaf: Leaf size of the dissection; see `jax_ls.dissection_tree`.
  """

  def __init__(self, m: int, n: int, leaf: int):
    self.m, self.n = m, n
    self.nodes: list[TreeNode] = list(dissection_tree(m, n, leaf).nodes())
    self.depth = max(node.depth for node in self.nodes)
    index = {id(node): i for i, node in enumerate(self.nodes)}
    # Leaves evenly spaced in elimination order, parents above their children.
    x, leaves = {}, 0
    for node in self.nodes:
      if node.children:
        x[id(node)] = np.mean([x[id(c)] for c in node.children])
      else:
        x[id(node)], leaves = float(leaves), leaves + 1
    self.xy = np.array([(x[id(node)], -node.depth) for node in self.nodes])
    self.edge_of: dict[
      int, int
    ] = {}  # child -> index of the edge to its parent
    self.edges = []
    for node in self.nodes:
      for child in node.children:
        self.edge_of[id(child)] = len(self.edges)
        self.edges.append((self.xy[index[id(node)]], self.xy[index[id(child)]]))
    sizes = np.array([len(node.separator) for node in self.nodes])
    self.marker_size = 40 + 260 * np.sqrt(sizes / sizes.max())
    # Rank of each unknown in the elimination order.
    self.rank = np.empty(m * n, dtype=int)
    self.rank[np.concatenate([node.separator for node in self.nodes])] = (
      np.arange(m * n)
    )

  def at_depth(self, d: int) -> list[TreeNode]:
    """The nodes at one depth.

    Args:
      d: Depth, 0 at the root.

    Returns:
      The nodes, in elimination order.
    """
    return [node for node in self.nodes if node.depth == d]


def _grid_image(ax, tree: DissectionTree):
  """An image of the grid with a hairline gap around every cell."""
  image = ax.imshow(
    np.ones((tree.m, tree.n, 3)),
    origin="lower",
    interpolation="nearest",
    extent=(-0.5, tree.n - 0.5, -0.5, tree.m - 0.5),
  )
  lines = [
    ((x - 0.5, -0.5), (x - 0.5, tree.m - 0.5)) for x in range(tree.n + 1)
  ]
  lines += [
    ((-0.5, y - 0.5), (tree.n - 0.5, y - 0.5)) for y in range(tree.m + 1)
  ]
  ax.add_collection(LineCollection(lines, colors=SURFACE, linewidths=1.6))
  ax.set_xlim(-0.5, tree.n - 0.5)
  ax.set_ylim(-0.5, tree.m - 0.5)
  ax.set_aspect("equal")
  ax.axis("off")
  return image


def _tree_artists(ax, tree: DissectionTree):
  """Edges and nodes of the tree, one row per depth."""
  edges = LineCollection(tree.edges, colors=EDGE, linewidths=1.2, zorder=1)
  ax.add_collection(edges)
  nodes = ax.scatter(
    tree.xy[:, 0],
    tree.xy[:, 1],
    s=tree.marker_size,
    c=[SURFACE],
    edgecolors=EDGE,
    linewidths=1.2,
    zorder=2,
  )
  ax.set_xlim(tree.xy[:, 0].min() - 0.8, tree.xy[:, 0].max() + 0.8)
  ax.set_ylim(-tree.depth - 0.6, 0.6)
  ax.set_yticks(-np.arange(tree.depth + 1))
  ax.set_yticklabels(
    [f"depth {d}" for d in range(tree.depth + 1)], fontsize=9, color=MUTED
  )
  ax.set_xticks([])
  ax.tick_params(length=0)
  for spine in ax.spines.values():
    spine.set_visible(False)
  return edges, nodes


def animate_dissection(
  tree: DissectionTree, path: str, fps: int = 12, dpi: int = 72
) -> str:
  """Separators sweep across their boxes, one depth at a time, beside the tree.

  Args:
    tree: The tree to draw.
    path: Where to write the GIF.
    fps: Frames per second.
    dpi: Resolution.

  Returns:
    `path`.
  """
  ramp = [rgb(c) for c in DEPTH_RAMP[: tree.depth + 1]]
  sweep, hold, final = 10, 6, 14
  frames = [
    (d, (i + 1) / sweep)
    for d in range(tree.depth + 1)
    for i in range(sweep + hold)
  ]
  frames += [(tree.depth, 1.0)] * final

  fig = plt.figure(figsize=(10.5, 5.0), facecolor=SURFACE)
  ax_grid = fig.add_axes((0.03, 0.07, 0.40, 0.70))
  ax_tree = fig.add_axes((0.50, 0.07, 0.48, 0.70))
  image = _grid_image(ax_grid, tree)
  _, nodes = _tree_artists(ax_tree, tree)
  ax_grid.set_title("the grid", fontsize=10, color=MUTED, loc="left")
  ax_tree.set_title(
    "the dissection tree (marker area ~ separator size)",
    fontsize=10,
    color=MUTED,
    loc="left",
  )

  def draw(i: int) -> None:
    """Draw frame `i`."""
    d, t = frames[i]
    cells = np.tile(rgb(SURFACE), (tree.m * tree.n, 1))
    faces = []
    for node in tree.nodes:
      if node.depth < d:
        cells[node.separator] = ramp[node.depth]
        faces.append(ramp[node.depth])
      elif node.depth == d:
        order = np.sort(node.separator)  # along the line, or row by row
        cells[order[: int(np.ceil(ease(t) * len(order)))]] = ramp[d]
        faces.append(mix(rgb(SURFACE), ramp[d], ease(t)))
      else:
        faces.append(rgb(SURFACE))
    image.set_data(cells.reshape(tree.m, tree.n, 3))
    nodes.set_facecolors(faces)
    k = len(tree.at_depth(d))
    if d == tree.depth:
      what = f"{k} leaf boxes, kept whole"
    else:
      what = f"{k} box{'es' if k > 1 else ''} cut by a one-cell separator"
    header(
      fig,
      f"Nested dissection, depth {d} of {tree.depth}",
      f"{what}. Each separator splits its box into two halves that no matrix "
      "entry couples.",
    )
    progress(fig, (i + 1) / len(frames))

  return save_gif(fig, draw, len(frames), path, fps, dpi)


def _step_colors(
  tree: DissectionTree, phase: str, d: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Colours of the cells, the tree nodes and the stored blocks at one step.

  Args:
    tree: The tree.
    phase: ``"up"`` (factorise and solve up the tree) or ``"down"``.
    d: The depth being processed.

  Returns:
    RGB arrays for the cells ``(N, 3)``, the nodes ``(n_nodes, 3)`` and the
    stored blocks ``(N, N, 3)``.
  """
  n_dof = tree.m * tree.n

  def done(node: TreeNode) -> bool:
    return node.depth > d if phase == "up" else node.depth < d

  cells = np.tile(rgb(PENDING), (n_dof, 1))
  for node in tree.nodes:
    if done(node):
      cells[node.separator] = rgb(DONE)
  current = tree.at_depth(d)
  for node in current:
    cells[node.boundary] = rgb(BOUNDARY)
  for node in current:
    cells[node.separator] = rgb(CURRENT)
  faces = np.array(
    [
      rgb(CURRENT)
      if node.depth == d
      else rgb(DONE)
      if done(node)
      else rgb(SURFACE)
      for node in tree.nodes
    ]
  )
  # Stored blocks: earlier levels muted, this level bright.  Up the tree they
  # appear as they are computed; down the tree they are all there already.
  blocks = np.tile(rgb(SURFACE), (n_dof, n_dof, 1))
  for node in tree.nodes:
    if phase == "up" and node.depth < d:
      continue
    weight = 1.0 if node.depth == d else 0.35
    s, b = tree.rank[node.separator], tree.rank[node.boundary]
    off_diagonal = mix(rgb(SURFACE), rgb(BOUNDARY), weight)
    blocks[np.ix_(b, s)] = off_diagonal
    blocks[np.ix_(s, b)] = off_diagonal
    blocks[np.ix_(s, s)] = mix(rgb(SURFACE), rgb(CURRENT), weight)
  return cells, faces, blocks


def animate_elimination(
  tree: DissectionTree, path: str, fps: int = 12, dpi: int = 64
) -> str:
  """Grid, tree and stored blocks, step by step up the tree and back down.

  Args:
    tree: The tree to draw.
    path: Where to write the GIF.
    fps: Frames per second.
    dpi: Resolution.

  Returns:
    `path`.
  """
  steps = [("up", d) for d in range(tree.depth, -1, -1)]
  steps += [("down", d) for d in range(tree.depth + 1)]
  colors = [_step_colors(tree, *step) for step in steps]
  n_dof = tree.m * tree.n
  start = (
    np.tile(rgb(PENDING), (n_dof, 1)),
    np.tile(rgb(SURFACE), (len(tree.nodes), 1)),
    np.tile(rgb(SURFACE), (n_dof, n_dof, 1)),
  )
  transition, hold, final = 6, 8, 12
  frames = [
    (k, (i + 1) / transition)
    for k in range(len(steps))
    for i in range(transition + hold)
  ]
  frames += [(len(steps) - 1, 1.0)] * final

  fig = plt.figure(figsize=(13.5, 5.2), facecolor=SURFACE)
  ax_grid = fig.add_axes((0.02, 0.15, 0.27, 0.62))
  ax_tree = fig.add_axes((0.33, 0.15, 0.36, 0.62))
  ax_blocks = fig.add_axes((0.72, 0.15, 0.27, 0.62))
  image = _grid_image(ax_grid, tree)
  edges, nodes = _tree_artists(ax_tree, tree)
  matrix = ax_blocks.imshow(start[2], interpolation="nearest")
  ax_blocks.set_xticks([])
  ax_blocks.set_yticks([])
  for spine in ax_blocks.spines.values():
    spine.set_color(EDGE)
  ax_grid.set_title("the grid", fontsize=10, color=MUTED, loc="left")
  ax_tree.set_title("the tree", fontsize=10, color=MUTED, loc="left")
  ax_blocks.set_title(
    "stored blocks, in elimination order", fontsize=10, color=MUTED, loc="left"
  )
  fig.legend(
    handles=[
      Patch(
        color=CURRENT, label="separators S of this depth / their inverse blocks"
      ),
      Patch(
        color=BOUNDARY, label="their boundaries B / lower and upper blocks"
      ),
      Patch(color=DONE, label="already eliminated (up) or solved (down)"),
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.04),
    ncol=3,
    frameon=False,
    fontsize=9.5,
  )

  def draw(i: int) -> None:
    """Draw frame `i`."""
    k, t = frames[i]
    phase, d = steps[k]
    before = colors[k - 1] if k else start
    cells, faces, blocks = (fade(b, a, t) for b, a in zip(before, colors[k]))
    image.set_data(cells.reshape(tree.m, tree.n, 3))
    nodes.set_facecolors(faces)
    matrix.set_data(blocks)
    # Highlight the tree edges along which information moves at this step.
    edge_colors = [EDGE] * len(tree.edges)
    widths = [1.2] * len(tree.edges)
    for node in tree.at_depth(d):
      if id(node) in tree.edge_of:
        edge_colors[tree.edge_of[id(node)]] = BOUNDARY
        widths[tree.edge_of[id(node)]] = 1.2 + 2.0 * ease(t)
    edges.set_colors(edge_colors)
    edges.set_linewidths(widths)
    count = len(tree.at_depth(d))
    fronts = "1 front" if count == 1 else f"{count} fronts at once"
    if phase == "up":
      header(
        fig,
        f"Factorise and solve up the tree: depth {d}",
        f"{fronts}: invert F_SS, store lower and upper, and send the Schur "
        "complement up to B.",
      )
    else:
      header(
        fig,
        f"Solve down the tree: depth {d}",
        f"{fronts}: x_S = z_S - upper x_B, with B solved by a shallower level.",
      )
    progress(fig, (i + 1) / len(frames))

  return save_gif(fig, draw, len(frames), path, fps, dpi)


# ---------------------------------------------------------------------------
# Inversions as they run.
# ---------------------------------------------------------------------------
def _frame_indices(n_iterates: int, max_frames: int) -> np.ndarray:
  """Evenly spaced iterates to show, always including the first and last."""
  count = min(n_iterates, max_frames)
  return np.unique(np.round(np.linspace(0, n_iterates - 1, count)).astype(int))


def _stage_normalised_loss(run: Trajectory) -> np.ndarray:
  """Each misfit divided by the misfit at the start of its frequency stage.

  The stages of a sweep minimise different objectives, so raw misfits are not
  comparable across them; relative to its own starting point, each stage's
  decay is.
  """
  start = {}
  for k, stage in enumerate(run.stage):
    start.setdefault(int(stage), run.loss[k])
  return np.array(
    [run.loss[k] / start[int(s)] for k, s in enumerate(run.stage)]
  )


def animate_reconstruction(
  domain: Domain,
  truth: Array,
  runs: dict[str, Trajectory],
  path: str,
  title: str = "Reconstruction",
  max_frames: int = 90,
  fps: int = 8,
  dpi: int = 64,
) -> str:
  """Recorded inversions side by side, with their error and misfit as they go.

  One panel shows the true contrast and one each run's current iterate, on a
  shared diverging colour scale; on the right, the relative model error and
  the misfit are drawn up to the current iteration.  Runs of different lengths
  are drawn against a common iteration count; a finished run holds its last
  iterate.  For a frequency sweep, stage changes are marked and the misfit is
  shown relative to the start of each stage, since every stage minimises its
  own objective.

  Args:
    domain: The grid.
    truth: The true contrast, flattened.
    runs: Label to `jax_ls.Trajectory`, from `jax_ls.trajectory` on a recorded
      result.  At most four.
    path: Where to write the GIF.
    title: First line of the frame header.
    max_frames: Longer runs are subsampled to this many frames, keeping the
      first and last iterates.
    fps: Frames per second.
    dpi: Resolution.

  Returns:
    `path`.

  Raises:
    ValueError: If there are no runs, or more than four.
  """
  if not 1 <= len(runs) <= len(SERIES):
    raise ValueError(f"between 1 and {len(SERIES)} runs, got {len(runs)}")
  labels = list(runs)
  truth = jnp.asarray(truth)
  errors = {label: run.relative_error(truth) for label, run in runs.items()}
  misfits = {label: _stage_normalised_loss(run) for label, run in runs.items()}
  n_iterates = max(run.nu.shape[0] for run in runs.values())
  shown = _frame_indices(n_iterates, max_frames)
  frames = list(shown) + [shown[-1]] * max(6, fps * 2)  # hold the end
  scale = 1.15 * float(jnp.max(jnp.abs(truth)))
  extent = (-domain.ax / 2, domain.ax / 2, -domain.ay / 2, domain.ay / 2)

  n_images = 1 + len(runs)
  fig = plt.figure(figsize=(3.3 * n_images + 4.6, 4.9), facecolor=SURFACE)
  image_width = 0.64 / n_images - 0.008
  axes_images = [
    fig.add_axes((0.02 + k * (image_width + 0.008), 0.12, image_width, 0.66))
    for k in range(n_images)
  ]
  ax_error = fig.add_axes((0.70, 0.50, 0.28, 0.30))
  ax_misfit = fig.add_axes((0.70, 0.12, 0.28, 0.30))

  pictures = []
  for ax, field in zip(
    axes_images, [truth] + [run.nu[0] for run in runs.values()]
  ):
    pictures.append(
      ax.imshow(
        domain.grid(field),
        extent=extent,
        origin="lower",
        cmap=CONTRAST_MAP,
        vmin=-scale,
        vmax=scale,
        interpolation="nearest",
      )
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
      spine.set_color(EDGE)
  axes_images[0].set_title(
    "true contrast", fontsize=10, color=MUTED, loc="left"
  )
  bar = fig.colorbar(
    pictures[0],
    ax=axes_images,
    orientation="horizontal",
    fraction=0.05,
    pad=0.04,
  )
  bar.ax.tick_params(labelsize=8, colors=MUTED)
  bar.set_label("contrast", fontsize=9, color=MUTED)
  bar.outline.set_edgecolor(EDGE)

  for ax, ylabel in (
    (ax_error, "relative model error"),
    (ax_misfit, "misfit / misfit at stage start"),
  ):
    ax.set_xlim(0, n_iterates - 1)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.grid(color=EDGE, linewidth=0.6)
    for side in ("top", "right"):
      ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
      ax.spines[side].set_color(EDGE)
  ax_misfit.set_xlabel("iteration", fontsize=9, color=MUTED)
  ax_error.set_ylim(
    0, 1.05 * max(1.0, max(float(e.max()) for e in errors.values()))
  )
  floor = min(float(m.min()) for m in misfits.values())
  ax_misfit.set_yscale("log")
  ax_misfit.set_ylim(max(floor, 1e-16) / 2, 2.0)

  # Stage changes of every sweep, marked on both curves and labelled with the
  # frequency the new stage starts.
  for run in runs.values():
    for k in np.flatnonzero(np.diff(run.stage)) + 1:
      for ax in (ax_error, ax_misfit):
        ax.axvline(
          k, color=MUTED, linewidth=0.9, linestyle=(0, (1.5, 2.5)), zorder=1
        )
      if run.omega:
        ax_error.annotate(
          f"omega {run.omega[int(run.stage[k])]:.0f}",
          xy=(k, 1.0),
          xycoords=("data", "axes fraction"),
          xytext=(2, -3),
          textcoords="offset points",
          fontsize=7.5,
          color=MUTED,
          rotation=90,
          ha="left",
          va="top",
        )

  lines, dots = {}, {}
  for color, label in zip(SERIES, labels):
    for ax, values in ((ax_error, errors[label]), (ax_misfit, misfits[label])):
      (line,) = ax.plot([], [], color=color, linewidth=2.0, label=label)
      (dot,) = ax.plot(
        [],
        [],
        "o",
        color=color,
        markersize=7,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
      )
      lines[label, ax] = (line, values)
      dots[label, ax] = dot
  # Middle right: clear of the stage labels along the top and of curves that
  # end near zero or above one.
  ax_error.legend(frameon=False, fontsize=9, loc="center right")

  def draw(i: int) -> None:
    """Draw frame `i`."""
    k = int(frames[i])
    for ax, label, color in zip(axes_images[1:], labels, SERIES):
      run = runs[label]
      at = min(k, run.nu.shape[0] - 1)
      pictures[1 + labels.index(label)].set_data(domain.grid(run.nu[at]))
      note = f"{label}, iteration {at}"
      if run.omega:
        stage = int(run.stage[at])
        note += f"\nomega = {run.omega[stage]:.1f} (stage {stage + 1}/{len(run.omega)})"
      ax.set_title(note, fontsize=10, color=color, loc="left")
    for (label, ax), (line, values) in lines.items():
      upto = min(k, len(values) - 1)
      line.set_data(np.arange(upto + 1), values[: upto + 1])
      dots[label, ax].set_data([upto], [values[upto]])
    summary = ", ".join(
      f"{label} {errors[label][min(k, len(errors[label]) - 1)]:.3f}"
      for label in labels
    )
    header(fig, title, f"Relative model error at iteration {k}: {summary}.")
    progress(fig, (i + 1) / len(frames))

  return save_gif(fig, draw, len(frames), path, fps, dpi)


__all__: Sequence[str] = (
  "DissectionTree",
  "animate_dissection",
  "animate_elimination",
  "animate_reconstruction",
  "save_gif",
)
