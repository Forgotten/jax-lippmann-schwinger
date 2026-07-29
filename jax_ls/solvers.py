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
