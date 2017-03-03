"""
Functions that are used across fitr modules
"""
import numpy as np

def softmax(x):
    xmax = np.max(x)
    return np.exp(x-xmax)/np.sum(np.exp(x-xmax))

def logsumexp(x):
    """
    Numerically stable logsumexp.
    """
    xmax = np.max(x)
    y = xmax + np.log(np.sum(np.exp(x-xmax)))
    return y
