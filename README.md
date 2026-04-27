# MPIDenseLUs

[![Build Status](https://github.com/johnomotani/MPIDenseLUs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/johnomotani/MPIDenseLUs.jl/actions/workflows/CI.yml?query=branch%3Amain)

Distributed parallelism Follows approach used by ScaLAPACK [J. Choi, et al.
"Design and implementation of the ScaLAPACK LU, QR, and Cholesky factorization
routines", Scientific Programming 5.3 (1996), pp. 173-184.], but with
'communication avoiding LU' pivoting from [L. Grigori, J. Demmel, and H. Xiang,
"CALU: a communication optimal LU factorization algorithm", SIAM Journal on
Matrix Analysis and Applications, 32 (2011), pp. 1317-1350].

Also uses shared-memory parallelism for matrix operations where the matrices
being operated on can be partitioned among multiple processes in `shared_comm`.

Includes scripts for benchmarking MPIDenseLUs against LAPACK/BLAS (via
LinearAlgebra), and ScaLAPACK (in Fortran).

The early development history of this solver is recorded in the
https://github.com/johnomotani/MPISchurComplements.jl repository. The solver
was then called `DenseLUs`, and was developed there up to
https://github.com/johnomotani/MPISchurComplements.jl/pull/23.
