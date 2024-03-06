import data
from mpi4py import MPI
import numpy as np
from os import environ
environ['OMP_NUM_THREADS'] = '1'


def mgs_sequential(A: np.array):
    """
    Sequentially compute the thin QR factorization of the input A A via
    modified Gram-Schmidt.

    Returns 'Q' (in-place) and 'R'.
    """
    n = A.shape[1]
    R = np.zeros((n, n), dtype=float)
    for k in range(n):
        R[k, k] = np.linalg.norm(A[:, k])
        A[:, k] /= R[k, k]
        R[k, k+1:n] = A[:, k].T @ A[:, k+1:n]
        A[:, k+1:n] -= np.outer(A[:, k], R[k, k+1:n])
    return A, R


def mgs_parallel(A: np.ndarray, rows: int, cols: int, comm: MPI.Comm):
    """
    Parallel compute the thin QR factorization of the input matrix A via
    modified Gram-Schmidt. Uses the processors from the given MPI communicator.

    Returns 'Q' (in-place) and 'R' on root processor.
    """
    # Setup matrices on all processors
    size = comm.Get_size()
    rank = comm.Get_rank()

    block_rows = rows // size
    R = np.zeros((cols, cols), dtype='float')
    submatrix = np.empty((block_rows, cols), dtype='float')

    comm.Scatterv(A, submatrix, root=0)

    # Iterate over columns
    for k in range(cols):
        # Locally compute as much as possibly
        v0 = submatrix[:, k].T @ submatrix[:, k:cols]
        # Sync norms
        v = comm.reduce(v0, op=MPI.SUM)
        if rank == 0:
            # Norm vectors
            v /= np.sqrt(v[0])
        else:
            v = np.empty((cols-k), dtype='float')
        comm.Bcast(v, root=0)

        # Finish computation locally
        submatrix[:, k] /= v[0]
        R[k, k:cols] = v

        submatrix[:, k+1:cols] -= np.outer(submatrix[:, k], R[k, k+1:cols])

    comm.Gatherv(submatrix, A)
    return A, R


if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Generata test matrix (W^1 from report)
    rows = 50000
    cols = 600
    A = None

    if rank == 0:
        A = data.W1(rows, cols)

    # Compute the QR factorization
    start_time = MPI.Wtime()
    if size == 1:
        Q, R = mgs_sequential(A)
    else:
        Q, R = mgs_parallel(A, rows, cols, comm)

    if rank == 0:
        end_time = MPI.Wtime()
        print("Runtime: ", end_time - start_time)
