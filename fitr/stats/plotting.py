import numpy as np 
from fitr.stats import linear_regression
import matplotlib.pyplot as plt 

def lrplot(X, 
           y):
    """ Plots linear regression model 
    
    Arguments: 

        X: `ndarray((nsamples,))`. Predictor variable 
        y: `ndarray((nsamples,))`. Response variable 

    Returns: 

        `matplotlib.pyplot.Figure`.  
    """
