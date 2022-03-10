import numpy as np
import PCSI.spline as sl


def test_func(x):
    return 1 / (1 + x ** 2)


points = list()
for i in range(-5, 6):
    points.append(np.array([i, test_func(i)]))

ret = sl.spline(points, sl.ConstraintType.DERIVATIVE1, m0=0.0147928994, mn=-0.0147928994)
print(ret)
