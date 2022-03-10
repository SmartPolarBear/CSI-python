import numpy as np
import sympy as sp

from typing import Final


def create_function(M, x: np.ndarray, y: np.ndarray, h: np.ndarray, N: Final):
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

    return X, sp.Piecewise(*formulas)
