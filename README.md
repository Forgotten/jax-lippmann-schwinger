# jax-lippmann-schwinger

This repository contains: 

- light weight solver for the Lippmann-Schwinger equation using JAX, and
- light weight solver for the inverse scattering problem using PDE-constrained optimization. 

In particular, we implement a time-harmonic wave solver using an integral formulation resulting in the Lippmann-Schwinger equation. The quadrature is implemented using the Greengard-Vico-Ferrando method, which is applied in quasi-linear time using FFT. The system is solved using GMRES (provided by jax).

In addition, we implement the near-field map, using the solver mentioned above. We implement the linearization of the near-field map (otherwise known as the Born approximation), together with the application of its adjoint. This allows us to compute the gradient of a l2 misfit loss efficiently. 

Finally, we use BFGS (also provided by jax) to solve the inverse problem, using the gradient computed via adjoint methods. 

## Organization 

This repository is organized as follows: 

- \jax_ls\: contains the code for the solver, derivatives and misfit
- \test\: contains several tests to check that the code generates the correct solution, and that all the adjoint test, and gradient tests. 
- \example\: contains a few examples of how to use the code
- \notebooks\: contains notebooks showcasing how the code solves the equation, and the minimization problem.

 
