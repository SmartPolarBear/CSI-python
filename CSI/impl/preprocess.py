import numpy as np

from typing import Final


def preprocess_points(points: list[np.ndarray]):
    x: np.ndarray = np.array([p[0] for p in points])
    y: np.ndarray = np.array([p[1] for p in points])
    h: np.ndarray = np.diff(x)
    return x, y, h


def preprocess_args(y: np.ndarray, h: np.ndarray, N: Final):
    alpha: np.ndarray = np.zeros(N + 1)
    beta: np.ndarray = np.zeros(N + 1)
    c: np.ndarray = np.zeros(N + 1)

    dy = np.diff(y)

    # by definition
    for j in range(1, N):
        alpha[j] = h[j - 1] / (h[j - 1] + h[j])
        beta[j] = 1 - alpha[j]
        c[j] = 6 * (1 / (h[j - 1] + h[j])) * ((dy[j] / h[j]) - (dy[j - 1] / h[j - 1]))

    return alpha, beta, c, dy
