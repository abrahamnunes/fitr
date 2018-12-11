import numpy as np


def canberra_distance(X, Y, p=1):
    """ Computes the Canberra distance between rows of X and Y (pairwise)
    
    $$
    d(X, Y) = \\sum_i^n \\frac{|x_{ij} - y_{ik}}{|x_{ij}| + |y_{ik}|}  \;\; \\forall j, k \\in \{1, 2, \ldots, n_f\}
    $$

    Arguments: 

        X: `ndarray((nsamples, nfeatures))`
        Y: `ndarray((nsamples, nfeatures))`

    Returns: 

        `ndarray((nfeatures, nfeatures))`

    """
    X, Y = pairwise_tile(X.T, Y.T)
    D = np.abs(Y - X)/(np.abs(X) + np.abs(Y))
    D = np.sum(D, 0) 
    return D

def distance(X, Y, metric='euclidean'):
    """ Computes the distance between rows of X and Y (pairwise)
    
    Arguments: 

        X: `ndarray((nsamples, nfeatures))`
        Y: `ndarray((nsamples, nfeatures))`
        metric: `str`. Type of distance to compute

    Returns: 

        `ndarray((nfeatures, nfeatures))`

    """
    if metric in ['euclidean', 'l2']:
        D = minkowski_distance(X, Y, p=1)
    elif metric == 'sqeuclidean':  
        D = minkowski_distance(X, Y, p=1)**2 
    elif metric in ['manhattan', 'l1']: 
        D = minkowski_distance(X, Y, p=0)
    elif metric == 'chebyshev': 
        D = minkowski_distance(X, Y, p=np.inf)
    elif metric == 'canberra':
        D = canberra_distance(X, Y)
    return D


def minkowski_distance(X, Y, p=1):
    """ Computes the Minkowski distance between rows of X and Y (pairwise)
    
    $$
    d(X, Y) = \\Big( \\sum_i^n |x_{ij} - y_{ik} |^p \\Big)^{\\frac{1}{p}} \;\; \\forall j, k \\in \{1, 2, \ldots, n_f\}
    $$

    Arguments: 

        X: `ndarray((nsamples, nfeatures))`
        Y: `ndarray((nsamples, nfeatures))`
        p: `float > 0`. 

    Returns: 

        `ndarray((nfeatures, nfeatures))`

    """
    d = pairwise_difference(X.T, Y.T)
    if p == np.inf: 
        D = np.max(np.abs(d), axis=0)
    elif p == -np.inf:
        D = np.min(np.abs(d), axis=0)
    else: 
        p = 2**p
        D = (np.sum(np.abs(d)**p,0)**(1/p))
    return D

def pairwise_difference(X, Y):
    """ Measures difference pairwise beween columns of `X` and `Y`

    Arguments: 

        X: `ndarray((nsamples, nfeatures))`
        Y: `ndarray((nsamples, nfeatures))`
        metric: `ndarray((nsamples, nsamples))`. Positive definite metric

    Returns: 
        
        `ndarray((nsamples, nfeatures, nfeatures))`

    """
    X, Y = pairwise_tile(X, Y)
    return Y-X

def pairwise_tile(X, Y):
    """ Tiles X and Y such that each row is compaired to each other row 

    Arguments: 

        X: `ndarray((nsamples, nfeatures))`
        Y: `ndarray((nsamples, nfeatures))`
        metric: `ndarray((nsamples, nsamples))`. Positive definite metric

    Returns: 
        
        X: `ndarray((nsamples, nfeatures, nfeatures))`
        Y: `ndarray((nsamples, nfeatures, nfeatures))`
        
    """
    n, m = X.shape
    X = np.tile(np.expand_dims(X, -1), [1, 1, m])
    Y = np.tile(np.expand_dims(Y, 1), [1, m, 1])
    return X, Y
