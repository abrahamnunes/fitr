import numpy as np
from fitr.utils import scale_data

def bic(log_prob, nparams, ntrials):
    """ Bayesian Information Criterion (BIC)

    Arguments:

        log_prob: Log probability
        nparams: Number of parameters in the model
        ntrials: Number of trials in the time series

    Returns:

        Scalar estimate of BIC.
    """
    return nparams * np.log(ntrials) - 2 * log_prob

def linear_correlation(X, Y):
    """ Linear correlation coefficient.

    Will compute the following formula

    $$
    \\rho = \\frac{\mathbf x^\\top \mathbf y}{\lVert \mathbf x \rVert \cdot \lVert \mathbf y \rVert}
    $$

    where each vector $\mathbf x$ and $\mathbf y$ are rows of the matrices $\mathbf X$ and $\mathbf Y$, respectively.

    Arguments:

        X: `ndarray((nsamples, nfeatures))` of dimension 1 or 2. If `X` is a 1D array, it will be converted to 2D prior to computation
        Y: `ndarray((nsamples, nfeatures))` of dimension 1 or 2. If `Y` is a 1D array, it will be converted to 2D prior to computation

    Returns:

        rho: `ndarray((nfeatures,))`. Correlation coefficient(s)

    TODO:

    - [ ] Create error raised when X and Y are not same dimension
    """
    # Reshape if necessary
    if X.ndim == 1 and Y.ndim == 1:
        X = X.reshape(-1, 1) - np.mean(X)
        Y = Y.reshape(-1, 1) - np.mean(Y)

    X = scale_data(X, axis=0, with_mean=True, with_var=False)
    Y = scale_data(Y, axis=0, with_mean=True, with_var=False)

    xnorm = np.linalg.norm(X, axis=0, ord=2)
    ynorm = np.linalg.norm(Y, axis=0, ord=2)
    rho = np.diag(X.T@Y)/(xnorm*ynorm)
    return rho

def lme(log_prob, nparams, hess_inv):
    """ Laplace approximation to the log model evidence

    Arguments:

        log_prob: Log probability
        nparams: Number of parameters in the model
        hess_inv: Hessian at the optimum (shape is $K \\times K$)

    Returns:

        Scalar approximation of the log model evidence
    """
    return log_prob + (nparams/2)*np.log(2*np.pi)-np.log(np.linalg.det(hess_inv))/2

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
