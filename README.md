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

## Usage

To LU factorize a matrix, existing in shared-memory on the 0'th shared-memory
block. The factorization can then be used to solve `A.x=b` where `b` is again
passed on the 0`th shared-memory block. The solution `x` is filled on every
shared-memory block. For example the following could be run on 4 MPI processes.
```julia
using LinearAlgebra
using MPI
using MPIDenseLUs

n = 1024
tile_size = 128
n_shared = 2

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
distributed_rank = rank ÷ n_shared
shared_comm = MPI.Comm_split(comm, distributed_rank, 0)
shared_rank = MPI.Comm_rank(shared_comm)
distributed_comm = MPI.Comm_split(comm, shared_rank == 0 ? 0 : nothing, 0)

local_win_store_float = MPI.Win[]
function allocate_shared_float(dims...)
    if shared_rank == 0
        dims_local = dims
    else
        dims_local = Tuple(0 for _ ∈ dims)
    end
    win, array_temp = MPI.Win_allocate_shared(Array{Float64}, dims_local,
                                              shared_comm)
    array = MPI.Win_shared_query(Array{Float64}, dims, win; rank=0)
    push!(local_win_store_float, win)
    if shared_rank == 0
        array .= NaN
    end
    MPI.Barrier(shared_comm)
    return array
end

local_win_store_int = MPI.Win[]
function allocate_shared_int(dims...)
    if shared_rank == 0
        dims_local = dims
    else
        dims_local = Tuple(0 for _ ∈ dims)
    end
    win, array_temp = MPI.Win_allocate_shared(Array{Int64}, dims_local,
                                              shared_comm)
    array = MPI.Win_shared_query(Array{Int64}, dims, win; rank=0)
    push!(local_win_store_int, win)
    if shared_rank == 0
        array .= typemin(Int64)
    end
    MPI.Barrier(shared_comm)
    return array
end

if distributed_rank == 0
    A = allocate_shared_float(n, n)
    b = allocate_shared_float(n)
    if shared_rank == 0
        A .= rand(n, n)
        b .= rand(n)
    end
    MPI.Barrier(shared_comm)
else
    A = nothing
    b = nothing
end
x = allocate_shared_float(n)

Alu = mpi_dense_lu(A, tile_size, comm, shared_comm, distributed_comm,
                   allocate_shared_float, allocate_shared_int)
ldiv!(x, Alu, b)

if distributed_rank == 0
    if shared_rank == 0
        A .= rand(n, n)
        b .= rand(n)
    end
    MPI.Barrier(shared_comm)
end

lu!(Alu, A)
ldiv!(x, Alu, b)

if local_win_store_float !== nothing
    # Free the MPI.Win objects, because if they are free'd by the garbage collector
    # it may cause an MPI error or hang.
    for w ∈ local_win_store_float
        MPI.free(w)
    end
    resize!(local_win_store_float, 0)
end
if local_win_store_int !== nothing
    # Free the MPI.Win objects, because if they are free'd by the garbage collector
    # it may cause an MPI error or hang.
    for w ∈ local_win_store_int
        MPI.free(w)
    end
    resize!(local_win_store_int, 0)
end
```
