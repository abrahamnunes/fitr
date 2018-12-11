import numpy as np

def entropy(p, axis=1, order=1):
    """ Computes entropy for a vector (or matrix) `p` 

    Arguments:

        p: `0 <= ndarray((n,m)) <=1`
        axis: `int`. The axis over which to compute entropies
        order: `float >= 0`. If `order=1` then we compute Shannon entropy

    Returns: 
        
        `ndarray(ndim=ndim(p)-1)` 

    """

    if order == 1: 
        H = np.sum(p*np.ma.log(p), axis=axis)
    else: 
        H = (1/(1-order))*np.ma.log(np.sum(p**order, axis=axis))
    return H


def lorenz_curve(X):
    """ Computes a (census) Lorenz curve for some input
    
    Arguments: 

        X: `ndarray((nsamples, nfeatures))`
    
    Returns: 

        P: `ndarray((nsamples, nfeatures))`. Cumulative population proportions 
        L: `ndarray((nsamples, nfeatures))`. Lorenz curves 

    """
    if np.ndim(X) == 1:
        X = np.reshape(X, [-1, 1])

    nsamples, nfeatures = X.shape
    X = np.sort(X, axis=0)
    C = np.cumsum(X, axis=0)
    S = np.sum(X, axis=0)
    S = np.tile(S, [nsamples, 1])
    L = C/S

    P = np.ones(nsamples)
    P = np.reshape(P, [-1, 1])
    P = np.tile(P, [1, nfeatures])
    P = np.cumsum(P, axis=0)
    Sp= np.tile(P[-1,:], [nsamples, 1])
    P = P/Sp
    return P, L

def gini(X, Y): 
    """ Computes the Gini coefficient for a Lorenz curve 
    
    Arguments: 

        X: `ndarray((nsamples, nfeatures))`. Cumulative population proportions
        Y: `ndarray((nsamples, nfeatures))`. Lorenz curves
    
    Returns: 

        `ndarray(nfeatures)`

    """
    return np.sum(X-Y, axis=0)/np.sum(X, axis=0)

def pietra(X, Y):
    """ Computes the Pietra coefficient for a Lorenz curve 
    
    Arguments: 

        X: `ndarray((nsamples, nfeatures))`. Cumulative population proportions
        Y: `ndarray((nsamples, nfeatures))`. Lorenz curves
    
    Returns: 

        `ndarray(nfeatures)`

    """
    if np.ndim(X) == 1: X = X.reshape(-1, 1)
    if np.ndim(Y) == 1: Y = Y.reshape(-1, 1)
    return np.max(X-Y, axis=0)
