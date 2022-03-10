import numpy as np
import sympy as sp

import matplotlib.pyplot as plt

import CSI.spline as sl


def test_func(x):
    return 1 / (1 + x ** 2)


points = list()
for i in range(-5, 6):
    points.append(np.array([i, test_func(i)]))

# X, spfunc1 = sl.spline(points, sl.ConstraintType.DERIVATIVE1, m0=0.0147928994, mn=-0.0147928994)
# X, spfunc2 = sl.spline(points, sl.ConstraintType.NATURAL, m0=0.0147928994, mn=-0.0147928994)
X, spfunc3 = sl.spline(points, sl.ConstraintType.NOT_A_KNOT)
# X, spfunc4 = sl.spline(points, sl.ConstraintType.PERIODIC, m0=0.0147928994, mn=-0.0147928994)

sp.plot(spfunc3, (X, -5, 5), backend='matplotlib')
