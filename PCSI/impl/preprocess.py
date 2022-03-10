import numpy as np


def preprocess(points: list[np.ndarray]):
    x: np.ndarray = np.array([p[0] for p in points])
    y: np.ndarray = np.array([p[1] for p in points])
    h: np.ndarray = np.diff(x)
    return x, y, h
