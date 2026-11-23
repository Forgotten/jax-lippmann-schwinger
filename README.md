# jax-lippmann-schwinger

This repository contains: 

- light weight solver for the Lippmann-Schwinger equation using JAX, and
- light weight solver for the inverse scattering problem using PDE-constrained optimization. 

In particular, we implement a time-harmonic wave solver using an integral formulation resulting in the Lippmann-Schwinger equation. The quadrature is implemented using the Greengard-Vico-Ferrando method, which is applied in quasi-linear time using FFT. The system is solved using GMRES (provided by jax).

In addition, we implement the near-field map, using the solver mentioned above. We implement the linearization of the near-field map (otherwise known as the Born approximation), together with the application of its adjoint. This allows us to compute the gradient of a l2 misfit loss efficiently. 

Finally, we use BFGS (also provided by JAX) to solve the inverse problem, using the gradient computed via adjoint methods.

## Installation

Requires Python 3.10 or newer.

### With `pip`, straight from GitHub

No clone needed — this is also what the Colab notebook uses:

```bash
pip install git+https://github.com/Forgotten/jax-lippmann-schwinger.git
```

### From a local clone

For development, install in editable mode so that changes take effect immediately:

```bash
git clone https://github.com/Forgotten/jax-lippmann-schwinger.git
cd jax-lippmann-schwinger
pip install -e .
```

Either way `pip` pulls in the dependencies (`numpy`, `scipy`, `jax`, `jaxlib`, `flax`). The
package is not published on PyPI, so `pip install jax-ls` will *not* work.

This installs the CPU build of JAX. To run on a GPU, install the matching JAX wheel
afterwards, following the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

### Running Tests

To run the unit tests, use `pytest`:

```bash
python -m pytest tests/ -v
```

## Conventions

The domain `[-ax/2, ax/2) x [-ay/2, ay/2)` is discretised with the *periodic* convention:
`n` points of spacing `hx = ax / n`, with no duplicated endpoint. This is the grid the
FFT-based convolution assumes, so `hx * hy` (available as `params.cell_area`) is the
quadrature weight consistent with the operator. Rectangular domains (`ax != ay`, `n != m`)
are supported.

## Differentiating the near-field map

JAX lets a function carry either a custom JVP or a custom VJP, not both, and the linearised
solver cannot be transposed automatically. There are therefore three entry points, and using
the wrong one gives a wrong derivative with no error:

| Function | Gives | Use with |
|---|---|---|
| `near_field_map` | `F[nu]`, matrix valued | plotting, generating data |
| `near_field_map_forward_mode` | `F` with a `custom_jvp` (Born) | `jax.jvp`, `jax.linearize` |
| `near_field_map_reverse_mode` | `F` with a `custom_vjp` (`J^T`) | `jax.grad`, `jax.vjp` |
| `born_linearization` | `J dnu`, explicitly | matrix-free solvers |
| `born_adjoint` | `J^H w`, the *Hermitian* adjoint | CGLS / Gauss-Newton — **not** `jax.grad` |

## Preconditioning

GMRES iteration counts grow with the frequency and the contrast. `jax_ls.sparsify` implements
L. Ying's sparsifying preconditioner ([arXiv:1408.4495](https://arxiv.org/abs/1408.4495)),
which keeps them at around ten:

```python
params = jax_ls.init_params(1.0, 1.0, 127, 127, 80.0, sparsify=True)
# or: params = jax_ls.with_sparsifying_preconditioner(params)
```

Every solve built on `params` is then right-preconditioned: `ls_solver`, the near-field map,
its derivatives and the inversions. Answers and derivatives are unchanged. At `n = 127`,
`omega = 80`, a smooth contrast of 4 goes from 1333 GMRES iterations to 11. The sparse LU it
needs runs on the host through `jax.pure_callback`, once per contrast, and is shared by every
forward and adjoint solve at that contrast. See
[`notebooks/sparsifying_preconditioner_notebook.ipynb`](notebooks/sparsifying_preconditioner_notebook.ipynb).

On accelerators, `sparsify_backend="block_inverse"` replaces SuperLU with
`jax_ls.nd_solver`, a nested-dissection solver written in `jax.numpy` that stores every
diagonal block already inverted. A solve is then batched GEMMs on the device, inside one
jitted program with GMRES and the gradients. The factorisation is cached across
every solve at one contrast: `linearize`, the near-field derivative rules and the loss's
gradient each factor once. See
[`notebooks/block_inverse_solver_notebook.ipynb`](notebooks/block_inverse_solver_notebook.ipynb).

By default the factorisation itself runs on the host CPU (`sparsify_factor_on="cpu"`, or
`factor_on="cpu"` in `with_sparsifying_preconditioner`): GPUs with weak double precision are
slowest at the batched complex128 LU. It uses NumPy and LAPACK in complex128 through
`jax.pure_callback`, still inside the jitted program, and copies the blocks to the device once
per contrast; every solve runs on the device as batched matrix products. `factor_on="device"`
factors in `jax.numpy` on the device instead, which is somewhat faster on a CPU-only machine.
On macOS, `factor_on="accelerate"` does the same with Apple Accelerate's LAPACK, factoring
each level's fronts in parallel. It helps when NumPy and SciPy are *not* already linked
against Accelerate, as in a default Anaconda install, which uses OpenBLAS. Check with
`numpy.show_config()` and `scipy.show_config()`. The pip wheels of NumPy 2.0+ and recent
SciPy on Apple Silicon with macOS 14+ already use Accelerate; there `factor_on="cpu"` runs on
Accelerate too, and the remaining gain is the parallel loop over fronts. The option needs
macOS 13.3+ and the Xcode Command Line Tools: a small C shim is compiled on first use and
cached in `~/Library/Caches/jax_ls` (or `$JAX_LS_CACHE_DIR`). On an M4 at n = 511, with
NumPy and SciPy on OpenBLAS, it factors in 3.8 s, against 7.6 s for `factor_on="cpu"` and
5.5 s for XLA on the CPU.

## Watching an inversion

The optimisers can keep every iterate: `QuasiNewton(record=True)`, `GaussNewton(record=True)`
or `frequency_sweep(..., record=True)`. `jax_ls.trajectory` turns a recorded result into the
sequence of iterates and misfits, and `jax_ls.animations` replays it as a GIF, beside the true
contrast and with the error and misfit drawn as they decay:

```python
from jax_ls import animations  # needs the viz extra: pip install "jax-ls[viz]"

result = jax_ls.minimize(params, data, nu0, jax_ls.QuasiNewton(maxiter=50, record=True))
animations.animate_reconstruction(
  domain, nu_true, {"BFGS": jax_ls.trajectory(result)}, "reconstruction.gif"
)
```

A recorded BFGS run uses SciPy's BFGS driver, with the loss and gradient still computed by
JAX, because `jax.scipy.optimize.minimize` runs as one compiled loop and exposes no
intermediate iterates. The two line searches differ, so a recorded run follows a slightly
different path. The same module draws the nested-dissection animations of the block-inverse
notebook.

## Organization 

This repository is organized as follows: 

- `jax_ls/`: contains the code for the solver, derivatives and misfit.
- `tests/`: contains several tests to check that the code generates the correct solution, including adjoint and gradient tests.
- `examples/`: contains a few examples of how to use the code.
- `notebooks/`: contains notebooks showcasing how the code solves the equation and the minimization problem.

 
