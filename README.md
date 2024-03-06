# Parallel Orthogonalization Techniques
In this project, the classical Gram-Schmidt (CGS), modified Gram-Schmidt (MGS)
and Tall-Skinny QR (TSQR) algorithms for orthogonalizing a set of vectors are
studied. The goal is to identify an efficient method that is numerically stable
and scales well in parallel. For details, see the project
[report](./report/report.pdf).

## Code

Python code for CGS, MGS and TSQR is provided in the `./code` folder. The
parallelization has been done via the Python implementation mpi4py of the
Message Passing Interface (MPI).

For sequential execution, run:
```console
python code/cgs.py
python code/msg.py
python code/tsqr.py
```

For parallel execution on 4 processors, run:
```console
mpiexec -n 4 python code/cgs.py
mpiexec -n 4 python code/msg.py
mpiexec -n 4 python code/tsqr.py
```
