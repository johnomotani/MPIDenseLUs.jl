Install dependencies. On a Ubuntu-like system, e.g.
```
sudo apt install libopenmpi-dev libopenblas-dev libscalapack-mpi-dev libhdf5-mpi-dev
```
(We don't actually need parallel hdf5, but will use the parallel HDF5 compiler
wrapper, so probably best to intsall the parallel version).

Compile the Fortran benchmark script
```
compile_benchmark.sh
```
