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
diagonal block already inverted. A solve is then batched GEMMs, and the factorisation, GMRES
and gradients stay in one jitted program on the device. The factorisation is cached across
every solve at one contrast: `linearize`, the near-field derivative rules and the loss's
gradient each factor once. See
[`notebooks/block_inverse_solver_notebook.ipynb`](notebooks/block_inverse_solver_notebook.ipynb).

## Organization 

This repository is organized as follows: 

- `jax_ls/`: contains the code for the solver, derivatives and misfit.
- `tests/`: contains several tests to check that the code generates the correct solution, including adjoint and gradient tests.
- `examples/`: contains a few examples of how to use the code.
- `notebooks/`: contains notebooks showcasing how the code solves the equation and the minimization problem.

 
