using MPI

include("julia-time-lu.jl")

MPI.Init()
temp_log_dir = tempname()
mkpath(temp_log_dir)
logfile = joinpath(temp_log_dir, "temp.log")
time_lu("matrices-and-rhs.h5", 1, nothing, 128, 1, 10, 1, 1, [128], logfile)
time_lu("matrices-and-rhs.h5", 1, 1, 128, 1, 10, 1, 1, [128], logfile)
MPI.Finalize()
