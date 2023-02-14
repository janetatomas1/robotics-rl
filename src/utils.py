import numpy as np


def distance(vec):
    d = 0

    for i in  range(1, len(vec)):
        d += np.linalg.norm(vec[i] - vec[i-1])

    return d
