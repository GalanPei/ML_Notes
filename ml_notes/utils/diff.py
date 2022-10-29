import numpy as np

def absolute_error(predict, truth):
    return np.abs(predict - truth)

def relative_error(predict, truth):
    return np.abs(predict - truth) / (np.abs(truth) + 1e-8)