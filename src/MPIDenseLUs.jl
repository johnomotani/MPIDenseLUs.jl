module MPIDenseLUs

export MPIDenseLU, mpi_dense_lu

using Combinatorics
using LinearAlgebra
using LinearAlgebra.BLAS: trsv!, trsm!, gemv!, gemm!
using MPI
using Primes
using TimerOutputs

import LinearAlgebra: lu!, ldiv!

@kwdef struct MPIDenseLU{T,Tmat,Tvec,Tintvec,Tfmp,Tslu,Tsync,Ttimer}
    m::Int64
    n::Int64
    row_permutation::Tintvec
    group_K::Int64
    group_L::Int64
    group_k::Int64
    group_l::Int64
    factorization_matrix_storage::Tmat
    factorization_matrix_parts::Tfmp
    factorization_matrix_parts_row_ranges::Vector{UnitRange{Int64}}
    factorization_matrix_parts_col_ranges::Vector{UnitRange{Int64}}
    factorization_locally_owned_rows::Vector{Int64}
    factorization_pivot_generation_distributed_tree_sizes::Vector{Int64}
    factorization_pivoting_buffer::Tvec
    factorization_local_left_panel_buffer::Vector{T}
    factorization_pivoting_reduction_buffer::Tmat
    factorization_pivoting_reduction_indices::Tintvec
    factorization_pivoting_reduction_indices_local::Vector{Int64}
    factorization_source_rows::Vector{Int64}
    factorization_locally_owned_swap_rows::Vector{Int64}
    factorization_row_swap_buffers::Tmat
    factorization_top_panel_pivots::Tintvec
    factorization_non_local_pivots::Tintvec
    factorization_top_panel_rows_to_send::Tintvec
    factorization_shared_lu::Tslu
    comm_requests::Vector{MPI.Request}
    my_L_tiles::Array{T,3}
    my_L_tile_row_ranges::Vector{UnitRange{Int64}}
    my_L_tile_col_ranges::Vector{UnitRange{Int64}}
    L_receive_requests::Vector{MPI.Request}
    L_send_requests::Vector{MPI.Request}
    my_U_tiles::Array{T,3}
    my_U_tile_row_ranges::Vector{UnitRange{Int64}}
    my_U_tile_col_ranges::Vector{UnitRange{Int64}}
    my_nonlocal_L_tile_list::Matrix{Int64}
    my_nonlocal_U_tile_list::Matrix{Int64}
    my_ldiv_tile_send_list::Matrix{Int64}
    my_local_L_tile_list::Matrix{Int64}
    my_local_U_tile_list::Matrix{Int64}
    my_diagonal_tile_list::Matrix{Int64}
    U_receive_requests::Vector{MPI.Request}
    U_send_requests::Vector{MPI.Request}
    diagonal_indices::Vector{Int64}
    new_column_triggers::Matrix{Int64}
    step_needs_synchronize_this_block::Tintvec
    vec_buffer1::Tvec
    vec_buffer2::Tvec
    L_rhs_update_buffer::Tvec
    U_rhs_update_buffer::Tvec
    tile_size::Int64
    n_tiles::Int64
    comm::MPI.Comm
    comm_rank::Int64
    comm_size::Int64
    shared_comm::MPI.Comm
    shared_comm_rank::Int64
    shared_comm_size::Int64
    distributed_comm::MPI.Comm
    distributed_comm_rank::Int64
    distributed_comm_size::Int64
    is_root::Bool
    synchronize_shared::Tsync
    check_lu::Bool
    timer::Ttimer
end

macro dlu_timeit(timer, name, expr)
    return quote
        if $(esc(timer)) === nothing
            $(esc(expr))
        else
            @timeit $(esc(timer)) $(esc(name)) $(esc(expr))
        end
    end
end

"""
    mpi_dense_lu(A::Union{AbstractMatrix,Nothing}, tile_size::Int64, comm::MPI.Comm,
                 shared_comm::MPI.Comm, distributed_comm::MPI.Comm,
                 allocate_shared_float::Function, allocate_shared_int::Function;
                 synchronize_shared::Union{Function,Nothing}=nothing,
                 distributed_block_rows::Union{Integer,Nothing}=nothing,
                 skip_factorization::Bool=false, check_lu::Bool=true,
                 timer::Union{TimerOutput,Nothing}=nothing)

The matrix `A` to be factorized must be passed only on the 0'th shared-memory block.
Factorization can be skipped by passing `skip_factorization=false`, but a matrix of the
correct size and type must be passed anyway.

`tile_size` is the tile size used for both factorization and matrix-solve.

MPI communicators are required: `comm` contains all the processes participating;
`shared_comm` contains the processes in each shared-memory block; `distributed_comm` is
required only on rank-0 of each shared-memory block and contains the rank-0 process of
each shared-memory block.

`allocate_shared_float` and `allocate_shared_int` are functions that allocate a
shared-memory array (shared by the processes in `shared_comm`) with float or integer type.

`synchronize_shared` can be passed a custom function to synchronize the processes in
`shared_comm`. By default `MPI.Barrier(shared_comm)` is used.

`distributed_block_rows` controls the layout of shared-memory blocks used for LU
factorization. If passed, it must be a factor of the size of `distributed_comm`, and the
blocks are laid out in `distributed_block_rows` rows and `MPI.Comm_size(distributed_comm)
÷ distributed_block_rows` columns. By default the number of rows is set to be as close to
`sqrt(MPI.Comm_size(distributed_comm))` as possible, with more rows than columns if the
numbers cannot be equal.

`check_lu=false` can be passed to skip checks that the matrix entries are finite.

`timer` can be passed a `TimerOutput` object to record timings.
"""
function mpi_dense_lu(A::Union{AbstractMatrix,Nothing}, tile_size::Int64, comm::MPI.Comm,
                      shared_comm::MPI.Comm, distributed_comm::MPI.Comm,
                      allocate_shared_float::Function, allocate_shared_int::Function;
                      synchronize_shared::Union{Function,Nothing}=nothing,
                      distributed_block_rows::Union{Integer,Nothing}=nothing,
                      skip_factorization::Bool=false, check_lu::Bool=true,
                      timer::Union{TimerOutput,Nothing}=nothing)
    @dlu_timeit timer "setup" begin
        if synchronize_shared === nothing
            synchronize_shared = ()->MPI.Barrier(shared_comm)
        end

        comm_rank = MPI.Comm_rank(comm)
        comm_size = MPI.Comm_size(comm)

        if comm_rank == 0
            datatype = eltype(A)
            MPI.bcast(datatype, comm; root=0)
            m, n = size(A)
            mref = Ref(m)
            nref = Ref(n)
            req1 = temp_Ibcast!(mref, comm; root=0)
            req2 = temp_Ibcast!(nref, comm; root=0)
            MPI.Waitall([req1, req2])
        else
            datatype = MPI.bcast(nothing, comm; root=0)
            mref = Ref(0)
            nref = Ref(0)
            req1 = temp_Ibcast!(mref, comm; root=0)
            req2 = temp_Ibcast!(nref, comm; root=0)
            MPI.Waitall([req1, req2])
            m = mref[]
            n = nref[]
        end
        if m != n
            error("Non-square matrices not supported in MPIDenseLU. Got ($m,$n).")
        end

        shared_comm_rank = MPI.Comm_rank(shared_comm)
        shared_comm_size = MPI.Comm_size(shared_comm)

        # distributed comm is only needed on the root process of each shared-memory block.
        if shared_comm_rank == 0
            distributed_comm_rank = Ref(MPI.Comm_rank(distributed_comm))
            distributed_comm_size = Ref(MPI.Comm_size(distributed_comm))
            # ...but need to know distributed_comm_size on all ranks for initialization.
        else
            distributed_comm_rank = Ref(-1)
            distributed_comm_size = Ref(-1)
        end
        MPI.Bcast!(distributed_comm_rank, shared_comm; root=0)
        MPI.Bcast!(distributed_comm_size, shared_comm; root=0)
        is_root = (shared_comm_rank == 0 && distributed_comm_rank[] == 0)

        if distributed_block_rows === nothing
            # Each block owns a set of (tile_size,tile_size) tiles in the full matrix - the
            # last row and column of tiles may be shorter/narrower. The tiles are distributed
            # in a block-cyclic pattern. Each block owns sub-tiles in the k'th row in each
            # group of K columns, and in the l'th column of each group of L columns. We choose
            # (abritrarily) to make L≤K.
            distributed_comm_size_factors =
                [prod(x) for x in
                 collect(unique(combinations(factor(Vector, distributed_comm_size[]))))]
            # Find the last factor ≤ sqrt(distributed_comm_size)
            factor_ind = findlast(x -> x≤sqrt(distributed_comm_size[]), distributed_comm_size_factors)
            group_L = distributed_comm_size_factors[factor_ind]
            group_K = distributed_comm_size[] ÷ group_L
        else
            if distributed_comm_size[] % distributed_block_rows != 0
                error("distributed_block_rows=$distributed_block_rows argument does not "
                      * "divide distributed_comm_size[]=$(distributed_comm_size[]).")
            end
            group_K = distributed_block_rows
            group_L = distributed_comm_size[] ÷ group_K
        end

        # setup_lu and setup_ldiv both return NamedTuples. All the entries in both those
        # NamedTuples are fields of the MPIDenseLU struct, which we splat into the
        # MPIDenseLU constructor to avoid having to type out long lists of variable names
        # repeatedly.
        lu_variables =
            setup_lu(m, n, tile_size, shared_comm, shared_comm_rank, shared_comm_size,
                     distributed_comm_rank[], distributed_comm_size[], datatype,
                     allocate_shared_float, allocate_shared_int, synchronize_shared,
                     group_K, group_L, timer)

        ldiv_variables =
            setup_ldiv(m, datatype, tile_size, comm, shared_comm, shared_comm_size,
                       shared_comm_rank, distributed_comm, distributed_comm_size[],
                       distributed_comm_rank[], is_root, allocate_shared_float,
                       allocate_shared_int, group_K, group_L)

        A_lu =  MPIDenseLU(; m, n, tile_size, comm, comm_rank, comm_size, shared_comm,
                           shared_comm_rank, shared_comm_size, distributed_comm,
                           distributed_comm_rank=distributed_comm_rank[],
                           distributed_comm_size=distributed_comm_size[], is_root,
                           synchronize_shared, check_lu, group_K, group_L,
                           lu_variables..., ldiv_variables..., timer)
    end

    if !skip_factorization
        lu!(A_lu, A)
    end

    synchronize_shared()

    return A_lu
end

include("lu.jl")

include("ldiv.jl")

end
