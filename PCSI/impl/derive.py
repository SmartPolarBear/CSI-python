import numpy as np
import sympy as sp

from typing import Final


def spline_impl_derive1(x: np.ndarray, y: np.ndarray, h: np.ndarray, m0: np.float, mn: np.float):
    n: Final = x.shape[0]

    alpha: np.ndarray = np.zeros(n - 1)
    beta: np.ndarray = np.zeros(n - 1)
    c: np.ndarray = np.zeros(n - 1)

    dy = np.diff(y)

    # by definition
    for j in range(1, n):
        alpha[j] = h[j - 1] / (h[j - 1] + h[j])
        beta[j] = 1 - alpha[j]
        c[j] = 6 * (1 / h[j - 1] + h[j]) * (dy[j] / h[j] - dy[j - 1] / h[j - 1])

    # by constraint
    alpha[n] = 1
    beta[0] = 1
    c[0] = (6.0 / h[0]) * (dy[0] / h[0] - m0)
    c[n] = (6.0 / h[n - 1]) * (dy[n - 1] / h[n - 1] + mn)

    a = np.zeros((n + 1, n + 1))

    a[0, 0] = 2
    a[0, 1] = beta[0]
    a[n, n - 1] = alpha[n]
    a[n, n] = 2
    for i in range(1, n):
        a[i, 0 + i - 1] = alpha[i]
        a[i, 1 + i - 1] = 2
        a[i, 2 + i - 1] = beta[i]
    pass

    m = np.ndarray(np.matrix(a).I * np.transpose(np.matrix(c)))

    formulas = []
    X = sp.Symbol('x')
    for j in range(0, n):
        formula = (m[j + 1] / (6 * h[j])) * (X - x[j]) ** 3 \
                  - (m[j] / (6 * h[j])) * (X - x[j + 1]) ** 3 \
                  + (y[j + 1] / h[j] - (h[j] * m[j + 1]) / 6) * (X - x[j]) \
                  - (y[j] / h[j] - (h[j] * m[j]) / 6) * (X - x[j + 1])
        formula = sp.expand(formula)
        formula = sp.collect(formula)
        formulas.append(formula)
    return formulas


def spline_impl_derive2(points: list[np.ndarray]):
    pass
