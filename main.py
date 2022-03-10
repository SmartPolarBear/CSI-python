import numpy as np
import sympy as sp

import matplotlib.pyplot as plt

import CSI.spline as sl


def test_func(x):
    return 1 / (1 + x ** 2)


x = [i for i in range(-5, 6)]
y = [test_func(i) for i in x]

# X, spfunc1 = sl.spline(points, sl.ConstraintType.DERIVATIVE1, m0=1, mn=0.6868)
# X, spfunc2 = sl.spline(x, y, sl.ConstraintType.DERIVATIVE2, M0=0, Mn=0)
X, spfunc3 = sl.spline(x, y, sl.ConstraintType.NOT_A_KNOT)

sp.plot(spfunc3, (X, -5, 5), backend='matplotlib')
