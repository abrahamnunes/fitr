import numpy as np


def log_loss(p, q):
    """ Computes log loss.

    $$
    \mathcal L = \mathbf p^\\top \log \mathbf q + (1-\mathbf p)^\\top \log (1 - \mathbf q)
    $$

    Arguments:

        p: Binary vector of true labels `ndarray((nsamples,))`
        q: Vector of estimates (between 0 and 1) of type `ndarray((nsamples,))`

    Returns:

        Scalar log loss
    """
    return p@np.log(q) + (1-p)@np.log(1-q)
