using LinearAlgebra

# Ensure BLAS only uses 1 thread, to avoid oversubscribing processes as we are probably
# already fully parallelised.
BLAS.set_num_threads(1)

using MPI
using StableRNGs
using Test

include("utils.jl")
include("mpi_dense_lu.jl")

function runtests()
    if !MPI.Initialized()
        MPI.Init()
    end
    @testset "MPIDenseLUs" begin
        mpi_dense_lu_tests()
    end
end
runtests()
