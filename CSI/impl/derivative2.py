import numpy as np
import sympy as sp

from typing import Final

from CSI.impl.preprocess import preprocess_args
from CSI.impl.thomas import thomas_solve
from CSI.impl.utils import create_function, create_matrix_A, calculate_coefficients


def spline_impl_derive2(x: np.ndarray, y: np.ndarray, h: np.ndarray, M0: np.float, Mn: np.float):
    N: Final = x.shape[0] - 1

    alpha, beta, c, dy = preprocess_args(y, h, N)

    # by constraint
    alpha[N] = 0
    beta[0] = 0
    c[0] = 2 * M0
    c[N] = 2 * Mn

    M: Final = thomas_solve(alpha[1:N + 1], 2 * np.ones(N + 1), beta[0:N], c)
    return calculate_coefficients(M, y, h, N)

