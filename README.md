# jax-lippmann-schwinger

This repository contains: 

- light weight solver for the Lippmann-Schwinger equation using JAX, and
- light weight solver for the inverse scattering problem using PDE-constrained optimization. 

In particular, we implement a time-harmonic wave solver using an integral formulation resulting in the Lippmann-Schwinger equation. The quadrature is implemented using the Greengard-Vico-Ferrando method, which is applied in quasi-linear time using FFT. The system is solved using GMRES (provided by jax).

In addition, we implement the near-field map, using the solver mentioned above. We implement the linearization of the near-field map (otherwise known as the Born approximation), together with the application of its adjoint. This allows us to compute the gradient of a l2 misfit loss efficiently. 

Finally, we use BFGS (also provided by JAX) to solve the inverse problem, using the gradient computed via adjoint methods.

## Installation

### From Source

Clone the repository and install it using `pip` in your environment:

```bash
git clone https://github.com/Forgotten/jax-lippmann-schwinger.git
cd jax-lippmann-schwinger
pip install -e .
```

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

## Organization 

This repository is organized as follows: 

- `jax_ls/`: contains the code for the solver, derivatives and misfit.
- `tests/`: contains several tests to check that the code generates the correct solution, including adjoint and gradient tests.
- `examples/`: contains a few examples of how to use the code.
- `notebooks/`: contains notebooks showcasing how the code solves the equation and the minimization problem.
- `docs/`: contains a study of the code ([`code_study.md`](docs/code_study.md)) and the design
  for a wide-band inverse solver ([`wideband_design.md`](docs/wideband_design.md)).

 
