
from os import environ
environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from mpi4py import MPI
import data


def tsqr(A: np.ndarray, rows: int, cols: int, comm: MPI.Comm):
    # Setup matrices on all processors
    size = comm.Get_size()
    rank = comm.Get_rank()

    block_rows = rows // size
    submatrix = np.empty((block_rows, cols), dtype='float')

    comm.Scatterv(A, submatrix, root=0)

    # Compute R
    tree_size = int(np.log2(size))
    Qs = np.empty(tree_size+1, dtype=np.matrix)
    P_new = rank

    Qs[0], R = np.linalg.qr(submatrix)
    for i in range(tree_size):
        if P_new % 2 == 0:
            m, n = R.shape
            R_temp = np.zeros((m, n), dtype='float')
            comm.Recv(R_temp, source=rank + i + 1, tag=44)
            Qs[i+1], R = np.linalg.qr(np.vstack((R, R_temp)))
        else:
            comm.Send(R, dest=rank - i - 1, tag=44)
            break
        P_new = (P_new + 1) // 2

    # Setup for backwards traversal
    active = False
    send = False
    if rank == 0:
        send = True

    # Compute Q
    for i in range(tree_size, 0, -1):
        if not active and rank % 2**(i-1) == 0:
            active = True

        if active and send:
            n = Qs[i-1].shape[1]
            comm.Send(Qs[i][n:2*n, :], dest=rank + i, tag=44)
            Qs[i-1] @= Qs[i][0:n, :]
        elif active:
            n = Qs[i-1].shape[1]
            Qs[i] = np.zeros((n, n), dtype='float')
            comm.Recv(Qs[i], source=rank - i, tag=44)
            Qs[i-1] @= Qs[i]
            send = True

    comm.Gatherv(Qs[0], A)
    return A, R


def tsqr_no_Q(A: np.ndarray, rows: int, cols: int, comm: MPI.Comm):
    # Setup matrices on all processors
    size = comm.Get_size()
    rank = comm.Get_rank()

    block_rows = rows // size
    submatrix = np.empty((block_rows, cols), dtype='float')

    comm.Scatterv(A, submatrix, root=0)

    Qs = np.empty(size, dtype=np.matrix)
    tree_size = int(np.log2(size))
    P_new = rank
    Qs[0], R = np.linalg.qr(submatrix)
    for i in range(tree_size):
        if P_new % 2 == 0:
            R_temp = np.zeros((cols, cols), dtype='float')
            comm.Recv(R_temp, source=rank + i + 1, tag=44)
            Qs[i+1], R = np.linalg.qr(np.vstack((R, R_temp)))
        else:
            comm.Send(R, dest=rank - i - 1, tag=44)
            break
        P_new = (P_new + 1) // 2

    comm.Gatherv(Qs[0], A)
    return A


# run using 4 processors:
# mpiexec -n 4 python tsqr.py
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

    # Compute the QR factorization (explicit Q)
    start_time = MPI.Wtime()
    Q, R = tsqr(A, rows, cols, comm)

    if rank == 0:
        end_time = MPI.Wtime()
        print("Runtime (explicit Q): ", end_time - start_time)

    # Compute the QR factorization (no Q)
    start_time = MPI.Wtime()
    Q = tsqr_no_Q(A, rows, cols, comm)

    if rank == 0:
        end_time = MPI.Wtime()
        print("Runtime (no Q): ", end_time - start_time)
