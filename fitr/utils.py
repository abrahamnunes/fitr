# -*- coding: utf-8 -*-
import numpy as np

def softmax(x):
    """ Computes the softmax function

    $$
    p(\mathbf{x}) = \\frac{e^{\mathbf{x} - \max_i x_i}}{\mathbf{1}^\\top e^{\mathbf{x} - \max_i x_i}}
    $$

    Arguments:
    
        x: Softmax logits (`ndarray((N,))`)

    Returns:

        Vector of probabilities of size `ndarray((N,))`
    """
    xmax = np.max(x)
    expx = np.exp(x-xmax)
    return expx/np.sum(expx)

def log_loss(p, q):
    """ Log-loss function.

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

def sigmoid(x, a_min=-10, a_max=10):
    """ Sigmoid function

    $$
    \sigma(x) = \\frac{1}{1 + e^{-x}}
    $$

    Arguments:

        x: Vector
        a_min: Lower bound at which to clip values of `x`
        a_max: Upper bound at which to clip values of `x`

    Returns:

        Vector between 0 and 1 of size `x.shape`
    """
    expnx = np.exp(-np.clip(x, a_min=a_min, a_max=a_max))
    return 1/(1+expnx)

def stable_exp(x, a_min=-10, a_max=10):
    """ Clipped exponential function

    Avoids overflow by clipping input values.

    Arguments:

        x: Vector of inputs
        a_min: Lower bound at which to clip values of `x`
        a_max: Upper bound at which to clip values of `x`

    Returns:

        Exponentiated values of `x`.
    """
    return np.exp(np.clip(x, a_min=a_min, a_max=a_max))

def logsumexp(x):
    """ Numerically stable logsumexp.

    Computed as follows:

    $$
    \max x + \log \sum_x e^{x - \max x}
    $$

    Arguments:

        x: `ndarray(shape=(nactions,))``

    Returns:

        `float`
    """
    xmax = np.max(x)
    y = xmax + np.log(np.sum(np.exp(x-xmax)))
    return y
