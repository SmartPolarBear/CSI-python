import numpy as np
import sympy as sp

from typing import Final

from CSI.impl.preprocess import preprocess_args
from CSI.impl.function import create_function


def spline_impl_not_a_knot(x: np.ndarray, y: np.ndarray, h: np.ndarray):
    N: Final = x.shape[0] - 1

    alpha, beta, c, dy = preprocess_args(y, h, N)

    # by constraint
    alpha[N] = -2
    beta[0] = -2
    c[0] = 0
    c[N] = 0

    a = np.zeros((N + 1, N + 1))

    a[0, 0] = 2
    a[0, 1] = beta[0]
    a[N, N - 1] = alpha[N]
    a[N, N] = 2
    for i in range(1, N):
        a[i, 0 + i - 1] = alpha[i]
        a[i, 1 + i - 1] = 2
        a[i, 2 + i - 1] = beta[i]

    M: Final = np.transpose(np.matrix(a).I.dot(np.transpose(np.matrix(c))))

    return create_function(M, x, y, h, N)
