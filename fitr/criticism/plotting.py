import numpy as np
import matplotlib.pyplot as plt
from fitr.metrics import linear_correlation

def make_diagline_range(x, padding=0.01, npoints=100):
    """ Creates coordinates for a diagonal line for actual vs. estimate plots.

    Arguments:

        x: `ndarray` (1D)
        padding: `float` (default=0.01). How much further to extend below and above range.
        npoints: `int`. Number of points on the line.

    Returns:

        range_min: `float`. Low value for range
        range_max: `float`. High value for range
        dline: `ndarray(shape=x.shape)`. Linear space between `range_min` and `range_max`

    """
    xmin  = np.min(x)
    xmax  = np.max(x)
    dline = np.linspace(xmin-padding*xmin, xmax+padding*xmax, npoints)
    return xmin, xmax, dline


def actual_estimate(y_true,
                    y_pred,
                    xlabel='Actual',
                    ylabel='Estimate',
                    corr=True,
                    figsize=None):
    """ Plots parameter estimates against the ground truth values.

    Arguments:

        y_true: `ndarray(nsamples)`. Vector of ground truth parameters
        y_pred: `ndarray(nsamples)`. Vector of parameter estimates
        xlabel: `str`. Label for x-axis
        ylabel: `str`. Label for y-axis
        corr: `bool`. Whether to plot correlation coefficient.
        figsize: `tuple`. Figure size (inches).

    Returns:

        `matplotlib.pyplot.Figure`


    """
    xmin, xmax, dline = make_diagline_range(np.hstack((y_true, y_pred)))

    rho = linear_correlation(y_true, y_pred)[0]

    if figsize is None: figsize = (6, 4)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dline, dline, c='k', ls='--', lw=1.5)
    ax.scatter(y_true, y_pred, c='k')
    ax.set_title(r'Pearson Correlation $\rho=%s$' %rho.round(3))
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    return fig
