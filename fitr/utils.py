import numpy as np

def softmax(x):
    """
        Computes softmax probability
    """
    return np.exp(x)/np.sum(np.exp(x))

def mnrandi(p):
    """
        Returns index of max value from a multinomial sample
    """
    return np.argmax(np.random.multinomial(1, p))

def logsumexp(x):
    """
    Protects against numerical overflow/underflow.

    Parameters
    ----------
    x : ndarray(K)
        Vector of K values

    Returns
    -------
    float

    References
    ----------
    Samuel Gershman's `mfit` package (https://github.com/sjgershm/mfit)
    """
    ym = np.max(x)
    yc = x - ym
    y  = ym + np.log(np.sum(np.exp(yc)))
    i  = np.argwhere(np.logical_not(np.isfinite(ym)))
    if np.size(i) != 0:
        y[i[0][0]] = ym[i[0][0]]
    return y

def paramtransform(params, paramrng, transformtype):
    """
    Transforms parameters between constrained and unconstrained spaces.

    Based on code from:
        Akam et al. (2015). PLoS Computational Biology, 11(12), 1–25.

    Parameters
    ----------
    params : ndarray
        1-D array of K parameter values, where K is the number of parameters
    paramrng : ndarray
        1-D array of strings specifying the domain of the K parameters. Acceptable values are 'unit' (interval [0, 1]), 'pos' (interval [0, +Inf]), or 'unc' (interval [-Inf, +Inf])
    transformtype: str
        Whether the transformation is from unconstrained to constrained space ('uc'), or from constrained to unconstrained space ('cu')

    Returns
    -------
    ndarray
        1-D array of the K transformed parameter values
    """
    K = np.size(paramrng)
    for k in range(0, K):
        if transformtype == 'uc':
            if paramrng[k] == 'unit':
                if params[k] < -16:
                    params[k] = -16
                params[k] = 1./(1 + np.exp(-params[k]))
            elif paramrng[k] == 'pos':
                if params[k] > 16:
                    params[k] = 16
                params[k] = np.exp(params[k])
            elif paramrng[k] == 'unc':
                params[k] = params[k]
            else:
                raise ValueError(paramrng[k] + ' is not a valid parameter range')
        elif transformtype == 'cu':
            if paramrng[k] == 'unit':
                params[k] = -np.log((1./params[k])-1)
            elif paramrng[k] == 'pos':
                params[k] = np.log(params[k])
            elif paramrng[k] == 'unc':
                params[k] = params[k]
            else:
                raise ValueError(paramrng[k] + ' is not a valid parameter range')
    return params
