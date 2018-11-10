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


    
