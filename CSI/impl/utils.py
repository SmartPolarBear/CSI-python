import numpy as np
import sympy as sp

from typing import Final


def create_function(M, x: np.ndarray, y: np.ndarray, h: np.ndarray, N: Final):
    sixes: Final = np.full(N, 6)
    m: Final = np.asarray(M.A1)

    co1 = m[1:] / (sixes * h)
    co2 = m[:N] / (-sixes * h)
    co3 = y[1:] / h - (h * m[1:]) / sixes
    co4 = (h * m[:N]) / sixes - (y[:N] / h)

    formulas = []
    X = sp.Symbol('x')
    for j in range(0, N):
        formula = co1[j] * (X - x[j]) ** 3 \
                  + co2[j] * (X - x[j + 1]) ** 3 \
                  + co3[j] * (X - x[j]) \
                  + co4[j] * (X - x[j + 1])
        formula = sp.expand(formula)
        formula = sp.collect(formula, syms=x)
        formulas.append((formula, sp.And(X >= x[j], X < x[j + 1])))

    return X, sp.Piecewise(*formulas)


def create_coefficient_matrix(r_top: np.ndarray, r_center: np.ndarray, r_bottom: np.ndarray):
    return np.diag(r_bottom, -1) + np.diag(r_center, 0) + np.diag(r_top, 1)
