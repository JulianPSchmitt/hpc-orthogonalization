import numpy as np


def f(mu, x):
    return np.sin(10 * (mu + x)) / (np.cos(100 * (mu - x)) + 1.1)


def W1(rows: int, cols: int):
    return np.fromfunction(lambda i, j: f(
        ((i - 1) / (rows - 1)), ((j - 1) / (cols - 1))),
        (rows, cols),
        dtype=float)
