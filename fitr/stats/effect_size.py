import numpy as np

def hedges_g(m1, s1, n1, m2, s2, n2, corrected=True):
    """ Computes Hedges' g

    Arguments:

        m1: `float`. Mean for sample 1
        s1: `float`. SD for sample 1
        n1: `float`. Number of observations in sample 1
        m2: `float`. Mean for sample 2
        s2: `float`. SD for sample 2
        n2: `float`. Number of observations in sample 2
        corrected: `bool`. If `True`, corrects for the bias in $g$

    Returns:

        `float`. Hedges' g
    """
    pooled_sd = np.sqrt(((n1-1)*(s1**2) + (n2-1)*(s2**2))/(n1 + n2 - 2))
    g = (m1 - m2)/pooled_sd
    a = n1 + n2 - 2
    J = gamma(a/2)/(np.sqrt(a/2)*gamma((a-1)/2))
    if corrected:
        g = J*g
    return g
