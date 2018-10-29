import numpy as np

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
