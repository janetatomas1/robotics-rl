import numpy as np


def distance(v):
    return sum([np.linalg.norm(v[i] - v[i-1]) for i in range(1, len(v))])


def angle(u, v):
    return np.arccos(np.dot(u, v) / (np.linalg.norm(u), np.linalg.norm(v)))


def angle_distance(v):
    return sum([np.linalg.norm(v[i] - v[i-1]) for i in range(1, len(v))])
