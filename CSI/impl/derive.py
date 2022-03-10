import numpy as np
import sympy as sp

from typing import Final


def spline_impl_derive1(x: np.ndarray, y: np.ndarray, h: np.ndarray, m0: np.float, mn: np.float):
    N: Final = x.shape[0] - 1

    alpha: np.ndarray = np.zeros(N + 1)
    beta: np.ndarray = np.zeros(N + 1)
    c: np.ndarray = np.zeros(N + 1)

    dy = np.diff(y)

    # by definition
    for j in range(1, N):
        alpha[j] = h[j - 1] / (h[j - 1] + h[j])
        beta[j] = 1 - alpha[j]
        c[j] = 6 * (1 / (h[j - 1] + h[j])) * ((dy[j] / h[j]) - (dy[j - 1] / h[j - 1]))

    # by constraint
    alpha[N] = 1
    beta[0] = 1
    c[0] = (6.0 / h[0]) * (dy[0] / h[0] - m0)
    c[N] = (6.0 / h[N - 1]) * (dy[N - 1] / h[N - 1] + mn)

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

    formulas = []
    X = sp.Symbol('x')
    for j in range(0, N):
        formula = (M[0, j + 1] / (6 * h[j])) * (X - x[j]) ** 3 \
                  - (M[0, j] / (6 * h[j])) * (X - x[j + 1]) ** 3 \
                  + (y[j + 1] / h[j] - (h[j] * M[0, j + 1]) / 6) * (X - x[j]) \
                  - (y[j] / h[j] - (h[j] * M[0, j]) / 6) * (X - x[j + 1])
        formula = sp.expand(formula)
        formula = sp.collect(formula, syms=x)
        formulas.append((formula, sp.And(X >= x[j], X < x[j + 1])))

    return X,sp.Piecewise(*formulas)


def spline_impl_derive2(points: list[np.ndarray]):
    pass
