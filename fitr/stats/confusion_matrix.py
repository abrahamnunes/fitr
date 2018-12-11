# -*- coding: utf-8 -*-
import numpy as np 
from fitr.utils import make_onehot

def confusion_matrix(ytrue, ypred):
    """ Creates a confusion matrix from some ground truth labels and predictions 

    Arguments: 
        
        ytrue: `ndarray(nsamples)`. Ground truth labels 
        ypred: `ndarray(nsamples)`. Predicted labels 

    Returns: 

        C: `ndarray((nlabels, nlabels))`. Confusion matrix 


    Example: 

    In the binary classification case, we may have the following: 

    ``` python 
    from fitr.stats import confusion_matrix
    
    ytrue = np.array([0, 1, 0, 1, 0])
    ypred = np.array([1, 1, 0, 1, 0])
    C = confusion_matrix(ytrue, ypred)
    tn, fp, fn, tp = C.flatten()
    ```
    """
    ytrue, _ = make_onehot(ytrue)
    ypred, _ = make_onehot(ypred)
    C = np.einsum('ij,ik->ijk', ytrue, ypred)
    C = np.sum(C, 0)
    return C

def cohen_kappa(ytrue, ypred, bayesian=False):
    """ Computes cohen's kappa 

    Arguments: 
        
        ytrue: `ndarray(nsamples)`. Ground truth labels 
        ypred: `ndarray(nsamples)`. Predicted labels 
        bayesian: `bool`. Computes posterior distribution over Cohen's Kappa score

    Returns: 

        kappa: `-1 <= float <= 1`. 
    """
    C = confusion_matrix(ytrue, ypred)
    po = np.trace(C)/np.sum(C)
    pe = np.dot(np.sum(C, 0)/np.sum(C), np.sum(C, 1)/np.sum(C))
    return (po-pe)/pe
