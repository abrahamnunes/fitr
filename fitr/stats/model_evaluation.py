import numpy as np

def aic(log_prob, K, n):
    """ Akaike Information Criterion (AIC)

    Arguments:

        log_prob: `float`. Log probability
        K: `int`. Number of parameters in the model

    Returns:

        `float`. Scalar estimate of AIC.
    """
    return 2*K - 2 * log_prob

def bic(log_prob, K, n):
    """ Bayesian Information Criterion (BIC)

    Arguments:

        log_prob: `float`. Log probability
        K: `int`. Number of parameters in the model
        n: `int`. Number of observations used to compute the `log_prob`

    Returns:

        `float`. Scalar estimate of BIC.
    """
    return K * np.log(n) - 2 * log_prob



def lme(log_prob, K, hess_inv):
    """ Laplace approximation to the log model evidence

    Arguments:

        log_prob: `float`. Log probability
        K: `int`. Number of parameters in the model
        hess_inv: `ndarray((K, K))`. Hessian at the optimum.

    Returns:

        Scalar approximation of the log model evidence
    """
    return log_prob + (K/2)*np.log(2*np.pi)-np.log(np.linalg.det(hess_inv))/2
