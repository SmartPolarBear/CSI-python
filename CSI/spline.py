import numpy as np
from enum import Enum

from CSI.impl.preprocess import preprocess_points

from CSI.impl.derivative1 import spline_impl_derive1
from CSI.impl.derivative2 import spline_impl_derive2
from CSI.impl.periodic import spline_impl_periodic


class ConstraintType(Enum):
    NOT_A_KNOT = 1
    NATURAL = 2,
    DERIVATIVE1 = 3
    DERIVATIVE2 = 4,
    PERIODIC = 5


def spline(points: list[np.ndarray], constraint_type: ConstraintType, **constrains):
    x, y, h = preprocess_points(points)

    if constraint_type == ConstraintType.DERIVATIVE1:
        return spline_impl_derive1(x, y, h, constrains["m0"], constrains["mn"])
    elif constraint_type == ConstraintType.DERIVATIVE2:
        return spline_impl_derive2(x, y, h, constrains["M0"], constrains["Mn"])
    elif constraint_type == ConstraintType.NATURAL:
        return spline_impl_derive2(x, y, h, M0=0, Mn=0)
    elif constraint_type == ConstraintType.PERIODIC:
        return spline_impl_periodic(x, y, h)
    else:
        raise RuntimeError("Invalid constraint type {}".format(constraint_type))
