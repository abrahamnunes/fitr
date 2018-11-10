import numpy as np
import fitr.utils as fu

def lorenz_curve(X):
    """ Computes a (census) Lorenz curve for a set of features. 
    
    Curves are computed along columns unless input is a vector.

    Attributes: 

        X: `ndarray((nsamples,nfeatures))`

    Returns:

       p:`ndarray((nsamples,nfeatures))`. Cumulative sample size 
       L: `ndarray((nsamples, nfeatures))`. Lorenz curve

    """ 
    if np.ndim(X) == 1:
        X = np.reshape(X, [-1, 1])

    nsamples = X.shape[0]
    N  = np.arange(nsamples).reshape(-1, 1)
    N  = np.tile(N, [1, X.shape[1]]) + 1
    Xs = np.sort(X, axis=0)
    Xt = fu.reduce_then_tile(Xs, axis=0, f=np.sum)
    Cs = np.cumsum(Xs, axis=0)/Xt
    p   = np.cumsum(N, axis=0)
    L = Cs/p
    return p, L

