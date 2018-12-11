# -*- coding: utf-8 -*-
import numpy as np 
import scipy.stats as ss

def mean_ci(X, axis=0, alpha=0.05):
    """ Computes the mean and two-tailed $\\alpha$ level confidence interval
    
    Arguments: 

        X: `ndarray((nsamples, nfeatures))`. Data to be summarized. If 
        axis: `int < ndim(X)`. Over which axis the statistics are to be computed
        alpha: `float`. Significance threshold 
    
    Returns: 
        
        m: `ndarray((nfeatures,))`. Means
        lci: `ndarray((nfeatures,))`. Lower bound of confidence interval 
        uci: `ndarray((nfeatures,))`. Upper bound of onfidence interval

    """
    a2 = alpha/2
    Z  = ss.norm.ppf(1-a2) 
    m = np.mean(X, axis=axis)
    sd = np.std(X, axis=axis)
    n = X.shape[axis]
    se = sd/np.sqrt(n)
    lci = m - Z*se
    uci = m + Z*se
    return m, lci, uci

        
