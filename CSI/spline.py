import numpy as np
from enum import Enum

from CSI.impl.preprocess import preprocess

from CSI.impl.derive import spline_impl_derive1


class ConstraintType(Enum):
    NOT_A_KNOT = 1
    NATURAL = 2,
    DERIVATIVE1 = 3
    DERIVATIVE2 = 4,
    PERIODIC = 5


def spline(points: list[np.ndarray], constraint_type: ConstraintType, **constrains):
    x, y, h = preprocess(points)

    if constraint_type == ConstraintType.DERIVATIVE1:
        return spline_impl_derive1(x, y, h, constrains["m0"], constrains["mn"])
    else:
        raise RuntimeError("Invalid constraint type {}".format(constraint_type))
