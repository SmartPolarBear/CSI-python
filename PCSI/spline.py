import numpy as np

from PCSI.impl.preprocess import preprocess

from PCSI.impl.derive import spline_impl_derive1


def spline(points: list[np.ndarray], constraint_type: str, **constrains):
    x, y, h = preprocess(points)
    if constraint_type == 'not-a-knot':
        pass
    elif constraint_type == 'natural':
        pass
    elif constraint_type == 'derivative1':
        return spline_impl_derive1(x, y, h, constrains["m0"], constrains["mn"])
    elif constraint_type == 'derivative2':
        pass
    elif constraint_type == 'periodic':
        pass
    else:
        raise RuntimeError("Invalid constraint type {}".format(constraint_type))
