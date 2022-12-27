
import numpy as np


def path_cost(vec):
    c = 0

    for i in  range(1, len(vec)):
        c += np.linalg.norm(vec[i] - vec[i-1])

    return c
