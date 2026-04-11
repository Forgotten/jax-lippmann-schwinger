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
/opt/anaconda3/envs/jax/bin/python -m pytest tests/ -v
```

## Organization 

This repository is organized as follows: 

- `jax_ls/`: contains the code for the solver, derivatives and misfit.
- `tests/`: contains several tests to check that the code generates the correct solution, including adjoint and gradient tests.
- `examples/`: contains a few examples of how to use the code.
- `notebooks/`: contains notebooks showcasing how the code solves the equation and the minimization problem.

 
