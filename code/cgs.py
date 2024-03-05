from mpi4py import MPI
import numpy as np
from os import environ
environ['OMP_NUM_THREADS'] = '1'  # deactivate BLAS multithreading


def f(mu, x):
    return np.sin(10 * (mu + x)) / (np.cos(100 * (mu - x)) + 1.1)


def cgs_sequential(A: np.ndarray):
    """
    Sequentially compute the thin QR factorization of the input matrix A.

    Returns 'Q' (in-place) and 'R'.
    """
    n = A.shape[1]
    R = np.zeros((n, n), dtype=float)
    R[0, 0] = np.linalg.norm(A[:, 0])
    A[:, 0] /= R[0, 0]
    for i in range(1, n):
        R[0:i, i] = A[:, 0:i].T @ A[:, i]
        A[:, i] -= A[:, 0:i] @ R[0:i, i]
        R[i, i] = np.linalg.norm(A[:, i])
        A[:, i] /= R[i, i]
    return A, R


def cgs_parallel(A: np.ndarray, rows: int, cols: int, comm: MPI.Comm):
    """
    Parallel compute the thin QR factorization of the input matrix A. Uses the
    processors from the given MPI communicator.

    Returns 'Q' (in-place) and 'R'.
    """
    # Setup matrices on all processors
    size = comm.Get_size()

    block_rows = rows // size
    R = np.zeros((cols, cols), dtype='float')
    submatrix = np.empty((block_rows, cols), dtype='float')

    comm.Scatterv(A, submatrix, root=0)

    # Iterate over columns
    for k in range(cols):
        if k > 0:
            # Locally compute as much as possibly and sync dot products
            R[0:k, k] = comm.allreduce(
                submatrix[:, 0:k].T @ submatrix[:, k], op=MPI.SUM)

            # Finish computation locally
            submatrix[:, k] -= submatrix[:, 0:k] @ R[0:k, k]

        # Norm the vector
        R[k, k] = np.sqrt(comm.allreduce(
            np.dot(submatrix[:, k], submatrix[:, k]), op=MPI.SUM))
        submatrix[:, k] /= R[k, k]

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
        A = np.fromfunction(lambda i, j: f(
            ((i - 1) / (rows - 1)), ((j - 1) / (cols - 1))),
            (rows, cols),
            dtype=float)

    # Compute the QR factorization
    start_time = MPI.Wtime()
    if size == 1:
        Q, R = cgs_sequential(A)
    else:
        Q, R = cgs_parallel(A, rows, cols, comm)

    if rank == 0:
        end_time = MPI.Wtime()
        print("Runtime: ", end_time - start_time)
