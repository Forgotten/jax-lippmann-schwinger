"""Smoke tests for `jax_ls.animations`: each animation writes a GIF.

What a frame looks like is checked by eye in the notebooks; these pin that the
functions run, that they write what they say, and that the frame counts follow
from their inputs.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
Image = pytest.importorskip("PIL.Image")

import jax_ls  # noqa: E402
from jax_ls import animations  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _frames(path) -> int:
  with Image.open(path) as image:
    return image.n_frames


class TestDissectionTree:
  def test_layout(self):
    tree = animations.DissectionTree(15, 15, 3)
    assert len(tree.nodes) == 31 and tree.depth == 4
    assert len(tree.edges) == len(tree.nodes) - 1
    # Every node sits one row below its parent.
    assert set(tree.xy[:, 1]) == {0, -1, -2, -3, -4}
    assert np.array_equal(np.sort(tree.rank), np.arange(225))


class TestNestedDissectionAnimations:
  def test_dissection_writes_one_frame_per_step(self, tmp_path):
    tree = animations.DissectionTree(9, 9, 3)
    path = animations.animate_dissection(tree, str(tmp_path / "d.gif"), dpi=30)
    assert _frames(path) == (tree.depth + 1) * 16 + 14

  def test_elimination_writes_one_frame_per_step(self, tmp_path):
    tree = animations.DissectionTree(9, 9, 3)
    path = animations.animate_elimination(tree, str(tmp_path / "e.gif"), dpi=30)
    assert _frames(path) == 2 * (tree.depth + 1) * 14 + 12

  def test_fade_never_mixes_two_hues(self):
    before = np.array([animations.rgb(animations.CURRENT)])
    after = np.array([animations.rgb(animations.BOUNDARY)])
    middle = animations.fade(before, after, 0.5)
    assert np.allclose(middle, animations.rgb(animations.PENDING))


class TestReconstructionAnimation:
  def _trajectory(self, n_iterates, n_dof, stages=None, omegas=()):
    nu = jnp.linspace(0.0, 1.0, n_iterates)[:, None] * jnp.ones((1, n_dof))
    return jax_ls.Trajectory(
      nu=nu,
      loss=np.geomspace(1.0, 1e-4, n_iterates),
      stage=np.zeros(n_iterates, dtype=int) if stages is None else stages,
      omega=omegas,
    )

  def test_runs_of_different_lengths(self, tmp_path):
    domain = jax_ls.Domain(1.0, 1.0, 12, 12)
    truth = jnp.ones(domain.n_dof)
    runs = {
      "sweep": self._trajectory(
        9, domain.n_dof, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), (1.0, 2.0, 3.0)
      ),
      "single": self._trajectory(5, domain.n_dof),
    }
    path = animations.animate_reconstruction(
      domain, truth, runs, str(tmp_path / "r.gif"), dpi=30, fps=4
    )
    # Nine iterates, then the last held for max(6, 2 fps) frames.
    assert _frames(path) == 9 + 8

  def test_long_runs_are_subsampled(self, tmp_path):
    domain = jax_ls.Domain(1.0, 1.0, 10, 10)
    runs = {"run": self._trajectory(300, domain.n_dof)}
    path = animations.animate_reconstruction(
      domain,
      jnp.ones(domain.n_dof),
      runs,
      str(tmp_path / "r.gif"),
      max_frames=20,
      dpi=30,
      fps=4,
    )
    assert _frames(path) == 20 + 8

  def test_run_count_is_bounded(self, tmp_path):
    domain = jax_ls.Domain(1.0, 1.0, 10, 10)
    with pytest.raises(ValueError):
      animations.animate_reconstruction(
        domain, jnp.ones(domain.n_dof), {}, str(tmp_path / "r.gif")
      )
