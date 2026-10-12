"""Configuration and convergence reporting for the linear solves.

The Lippmann-Schwinger operator ``A = -I + omega^2 diag(nu) G`` is square and
non-Hermitian, so GMRES is the right tool and there is nothing to replace.  What
this module adds is *control* and *visibility*:

* `GmresConfig` exposes ``tol``, ``atol``, ``restart``, ``maxiter`` and the
  initial guess, all of which used to be pinned at their defaults;
* `SolveInfo` reports whether the solve actually converged.

Why the residual is measured rather than read off
-------------------------------------------------
``jax.scipy.sparse.linalg.gmres`` returns an ``info`` value that is **always
0**, whether or not the solve converged: with ``restart=5, maxiter=2`` on a
contrast of 2.0 it returns ``info=0`` alongside a relative residual of 3.4e-4.
It carries no information.  So `solve_with_info` computes ``||A x - b|| / ||b||``
itself.  That costs one extra operator application on top of the twenty to
thirty a solve typically takes -- a few percent, paid only when asked for.

This matters because the failure is silent and contrast-driven: measured, the
solve reaches 3.4e-15 at ``nu = 0.1`` but only 1.3e-6 at ``nu = 4`` while taking
eight times as long, and an inversion sweep drives ``nu`` from zero up to the
true contrast.
"""

import jax
import jax.numpy as jnp

from flax import struct

Array = jax.Array


@struct.dataclass
class GmresConfig:
  """Settings for the GMRES solves.

  Every field is static: they are Python values handed to GMRES, not data, so
  changing one is a recompile rather than a new trace input.

  Attributes:
    tol: Relative convergence tolerance.
    atol: Absolute convergence tolerance.
    restart: Size of the Krylov subspace before restarting.
    maxiter: Maximum number of restarts; None lets GMRES choose.
    solve_method: How the inner least-squares problem is solved, "batched" or
      "incremental".  Unrelated to multiple right-hand sides, which are handled
      by `vmap`.
  """

  tol: float = struct.field(pytree_node=False, default=1e-5)
  atol: float = struct.field(pytree_node=False, default=0.0)
  restart: int = struct.field(pytree_node=False, default=20)
  maxiter: int | None = struct.field(pytree_node=False, default=None)
  solve_method: str = struct.field(pytree_node=False, default="batched")


@struct.dataclass
class SolveInfo:
  """The measured outcome of a linear solve.

  Attributes:
    relative_residual: ``||A x - b|| / ||b||``, computed after the fact.
    converged: Whether that residual meets the configured tolerance.
  """

  relative_residual: Array
  converged: Array

  def __str__(self) -> str:
    status = "converged" if bool(self.converged) else "NOT CONVERGED"
    return f"{status} (relative residual {float(self.relative_residual):.2e})"


def solve(
  operator, rhs: Array, config: GmresConfig, x0: Array | None = None
) -> Array:
  """Solve ``operator(x) = rhs`` with GMRES.

  Args:
    operator: A callable applying the linear operator.
    rhs: The right-hand side.
    config: Solver settings.
    x0: Optional initial guess; a natural warm start when solving a sequence of
      related systems.

  Returns:
    The solution.
  """
  solution, _ = jax.scipy.sparse.linalg.gmres(
    operator,
    rhs,
    x0=x0,
    tol=config.tol,
    atol=config.atol,
    restart=config.restart,
    maxiter=config.maxiter,
    solve_method=config.solve_method,
  )
  return solution


def solve_with_info(
  operator, rhs: Array, config: GmresConfig, x0: Array | None = None
) -> tuple[Array, SolveInfo]:
  """Solve ``operator(x) = rhs`` and report whether it converged.

  Costs one extra operator application over `solve`, because the residual is
  measured rather than taken from GMRES -- see the module docstring.

  Args:
    operator: A callable applying the linear operator.
    rhs: The right-hand side.
    config: Solver settings.
    x0: Optional initial guess.

  Returns:
    A tuple of (solution, `SolveInfo`).
  """
  solution = solve(operator, rhs, config, x0)

  rhs_norm = jnp.linalg.norm(rhs)
  residual = jnp.linalg.norm(operator(solution) - rhs)
  # A zero right-hand side has a zero solution; call that converged rather than
  # dividing by zero.
  relative = jnp.where(
    rhs_norm > 0, residual / jnp.where(rhs_norm > 0, rhs_norm, 1.0), 0.0
  )

  return solution, SolveInfo(
    relative_residual=relative,
    converged=residual <= jnp.maximum(config.tol * rhs_norm, config.atol),
  )


# ---------------------------------------------------------------------------
# Block GMRES: all right-hand sides in one Krylov space.
# ---------------------------------------------------------------------------
@struct.dataclass
class BlockGmresConfig:
  """Settings for block GMRES, which solves ``A X = B`` for all columns at once.

  Where it helps, and where it does not
  -------------------------------------
  A block Krylov space ``span{B, AB, ..., A^(m-1) B}`` has dimension up to
  ``m * p`` for ``p`` right-hand sides, and contains each single-vector space
  ``span{b_s, A b_s, ...}``, so ``m`` block steps are never worse than ``m``
  single-vector steps and are usually much better *per operator application*.
  Measured on this code it does reach a given residual in roughly half the
  applications per right-hand side that `solve` needs.

  It is nonetheless **slower here on CPU**, and the reason is structural.
  Orthogonalising against the accumulated basis costs ``O(m^2 p^2 n)`` while the
  FFT operator costs only ``O(m p n log n)``, so the dense work outgrows the
  physics as ``m p / log n_padded`` -- a factor of about 43 at ``n = 128`` with
  64 sources.  Measured there, one cycle at ``m = 12`` spent 33% of its time in
  the operator, 32% orthogonalising against an 832-vector basis, and the rest in
  QR and the block least-squares.

  There is also no redundancy to exploit between the right-hand sides, and that
  is by construction rather than bad luck: the ``O(omega)`` sampling rule
  (`jax_ls.inversion.FrequencySchedule`) chooses the number of sources to match
  the degrees of freedom of the scattered field, so the block ``B`` comes out
  numerically *full rank* -- its singular values decay to only 0.68 of the
  largest at 64 sources.  Low-rank compression of ``B`` is unavailable for the
  same reason.

  Two caveats keep this worth having.  The measurement is CPU-only, and the
  overhead is BLAS-3 dense algebra that an accelerator does far more cheaply
  relative to a bandwidth-bound FFT, so the verdict may invert on a GPU.  And
  the argument above turns on the operator being cheap; with a costlier one
  (three dimensions, or a nested solve) the balance shifts back.

  Attributes:
    n_steps: Block-Arnoldi steps per cycle, the ``m`` above.  The basis holds
      ``(m + 1) * p`` vectors, so memory grows linearly in this.
    cycles: Restart cycles.
    reorthogonalise: Apply a second Gram-Schmidt pass.  Block Arnoldi loses
      orthogonality quickly, so this defaults to on.
  """

  n_steps: int = struct.field(pytree_node=False, default=12)
  cycles: int = struct.field(pytree_node=False, default=1)
  reorthogonalise: bool = struct.field(pytree_node=False, default=True)


def _block_gmres_cycle(
  operator, rhs: Array, x0: Array, config: BlockGmresConfig
) -> Array:
  """One restart cycle of block GMRES."""
  n, p = rhs.shape
  m = config.n_steps
  width = (m + 1) * p

  block_residual = rhs - operator(x0)
  first, scale = jnp.linalg.qr(block_residual)

  basis = jnp.zeros((n, width), dtype=rhs.dtype).at[:, :p].set(first)
  hessenberg = jnp.zeros((width, m * p), dtype=rhs.dtype)

  def step(j, carry):
    basis, hessenberg = carry
    current = jax.lax.dynamic_slice(basis, (0, j * p), (n, p))
    mapped = operator(current)

    # Orthogonalise against the whole allocated basis, masking the columns not
    # filled yet; that keeps every shape static, which is what lets this run
    # inside `fori_loop`.
    filled = jnp.arange(width) < (j + 1) * p
    for _ in range(2 if config.reorthogonalise else 1):
      overlap = jnp.where(filled[:, None], basis.conj().T @ mapped, 0.0)
      mapped = mapped - basis @ overlap
      hessenberg = jax.lax.dynamic_update_slice(
        hessenberg,
        jax.lax.dynamic_slice(hessenberg, (0, j * p), (width, p)) + overlap,
        (0, j * p),
      )

    nxt, residual_block = jnp.linalg.qr(mapped)
    basis = jax.lax.dynamic_update_slice(basis, nxt, (0, (j + 1) * p))
    hessenberg = jax.lax.dynamic_update_slice(
      hessenberg, residual_block, ((j + 1) * p, j * p)
    )
    return basis, hessenberg

  basis, hessenberg = jax.lax.fori_loop(0, m, step, (basis, hessenberg))

  target = jnp.zeros((width, p), dtype=rhs.dtype).at[:p, :].set(scale)
  coefficients = jnp.linalg.lstsq(hessenberg, target, rcond=None)[0]
  return x0 + basis[:, : m * p] @ coefficients


def block_solve(
  operator,
  rhs: Array,
  config: BlockGmresConfig = BlockGmresConfig(),
  x0: Array | None = None,
) -> Array:
  """Solve ``operator(X) = rhs`` for every column of `rhs` simultaneously.

  Args:
    operator: Applies the linear operator to a whole ``(n, p)`` block.
    rhs: Right-hand sides, of shape ``(n, p)``.
    config: Block solver settings.
    x0: Optional initial guess of the same shape.

  Returns:
    The solution block, of shape ``(n, p)``.
  """
  solution = jnp.zeros_like(rhs) if x0 is None else x0
  for _ in range(config.cycles):
    solution = _block_gmres_cycle(operator, rhs, solution, config)
  return solution


def block_solve_with_info(
  operator,
  rhs: Array,
  config: BlockGmresConfig = BlockGmresConfig(),
  x0: Array | None = None,
) -> tuple[Array, SolveInfo]:
  """Solve a block system and report the Frobenius-norm relative residual.

  Args:
    operator: Applies the linear operator to a whole ``(n, p)`` block.
    rhs: Right-hand sides, of shape ``(n, p)``.
    config: Block solver settings.
    x0: Optional initial guess.

  Returns:
    A tuple of (solution block, `SolveInfo`).
  """
  solution = block_solve(operator, rhs, config, x0)

  rhs_norm = jnp.linalg.norm(rhs)
  residual = jnp.linalg.norm(operator(solution) - rhs)
  relative = jnp.where(
    rhs_norm > 0, residual / jnp.where(rhs_norm > 0, rhs_norm, 1.0), 0.0
  )
  return solution, SolveInfo(
    relative_residual=relative, converged=relative <= GmresConfig().tol
  )
