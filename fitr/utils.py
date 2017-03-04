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

def trans_UC(values_U, rng):
    'Transform parameters from unconstrained to constrained space.'
    if rng[0] == 'all_unc':
        return values_U
    values_T = []
    for value, rng in zip(values_U, rng):
        if rng   == 'unit':  # Range: 0 - 1.
            if value < -16.:
                value = -16.
            values_T.append(1./(1. + np.exp(-value)))  # Don't allow values smaller than 1e-
        elif rng   == 'half':  # Range: 0 - 0.5
            if value < -16.:
                value = -16.
            values_T.append(0.5/(1. + np.exp(-value)))  # Don't allow values smaller than 1e-7
        elif rng == 'pos':  # Range: 0 - inf
            if value > 16.:
                value = 16.
            values_T.append(np.exp(value))  # Don't allow values bigger than ~ 1e7.
        elif rng == 'unc': # Range: - inf - inf.
            values_T.append(value)
    return np.array(values_T)

def BIC(loglik, nparams, nsteps):
    """
    Calculates Bayesian information criterion
    """
    return nparams*np.log(nsteps) - 2*loglik

def AIC(nparams, loglik):
    """
    Calculates Aikake information criterion
    """
    return 2*nparams - 2*loglik

def LME(logpost, nparams, hessian):
    """
    Calculates log-model-evidence (LME)
    """
    return logpost + (nparams/2)*np.log(2*np.pi)-np.log(np.linalg.det(hessian))/2
