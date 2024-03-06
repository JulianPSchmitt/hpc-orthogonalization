import numpy as np


def f(mu, x):
    """
    Function 'f' (see report).
    """
    return np.sin(10 * (mu + x)) / (np.cos(100 * (mu - x)) + 1.1)


def W1(rows: int, cols: int):
    """
    Generate matrix 'W^1' (see report).
    """
    return np.fromfunction(lambda i, j: f(
        ((i - 1) / (rows - 1)), ((j - 1) / (cols - 1))),
        (rows, cols),
        dtype=float)
