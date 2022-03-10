import numpy as np
import sympy as sp

from typing import Final

from CSI.impl.preprocess import preprocess_args
from CSI.impl.utils import create_function, create_coefficient_matrix


def spline_impl_derive2(x: np.ndarray, y: np.ndarray, h: np.ndarray, M0: np.float, Mn: np.float):
    N: Final = x.shape[0] - 1

    alpha, beta, c, dy = preprocess_args(y, h, N)

    # by constraint
    alpha[N] = 0
    beta[0] = 0
    c[0] = 2 * M0
    c[N] = 2 * Mn

    a = create_coefficient_matrix(beta[0:N], 2 * np.ones(N + 1), alpha[1:N + 1])

    M: Final = np.transpose(np.matrix(a).I.dot(np.transpose(np.matrix(c))))

    return create_function(M, x, y, h, N)
