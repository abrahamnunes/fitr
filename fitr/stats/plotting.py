import numpy as np 
import scipy.stats as ss
from fitr.stats import linear_regression
import matplotlib.pyplot as plt 

def lrplot(x, 
           y, 
           title='',
           xlabel=r'$x$',
           ylabel=r'$y$',
           color='k',
           alpha=1.,
           figsize=None, 
           stat_pos=None):
    """ Plots linear regression model 
    
    Arguments: 

        x: `ndarray((nsamples,))`. Predictor variable 
        y: `ndarray((nsamples,))`. Response variable 
        title: `str`. Plot title
        xlabel: `str`. X-axis label
        ylabel: `str`. y-axis label
        color: `str`. Marker colors 
        alpha: `0 <= float <= 1`. Transparency of points
        figsize: `(width, height)` or `None`. 
        stat_pos: `(x, y)` or `None`. Coordinates at which to place the statistic values

    Returns: 

        `matplotlib.pyplot.Figure`.  
    """
    if figsize is None: 
        figsize = (5, 3)
    res = linear_regression(x, y)
    xmin, xmax = np.min(x), np.max(x)
    xspan = xmax-xmin
    xrng = np.linspace(xmin-0.02*xspan, xmax+0.02*xspan, 100).reshape(-1, 1)
    Xpred = np.hstack((np.ones_like(xrng), xrng))
    ypred = np.einsum('ij,j->i', Xpred, res.coef)
    ymin, ymax = np.min(ypred), np.max(ypred)
    yspan = ymax-ymin

    if stat_pos is None: 
        if res.coef[-1] > 0:
            stat_pos = [xmin+0.01*xspan, ymax-0.01*yspan]
        else:
            stat_pos = [xmin+0.01*xspan, ymin+0.01*yspan]

    if res.pcoef[1] < 0.001:
        pcoef = 'p_{coef}<0.001'
    else: 
        pcoef ='p_{coef}=%s' %res.pcoef[1].round(3)

    if res.pmodel < 0.001:
        pmodel = 'p_{model}<0.001'
    else: 
        pmodel = 'p_{model}=%s' %np.round(res.pmodel, 3)

    r2adj=res.R2adj.round(2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.scatter(x, y, c=color, alpha=alpha)
    ax.plot(xrng, ypred, c=color, lw=1.5)
    ax.text(stat_pos[0], stat_pos[1],
            s=r'$R^2_{Adj}=%s$, $%s$, $%s$' %(r2adj,pcoef,pmodel), 
            bbox=dict(facecolor='white', alpha=0.4))

    return fig
    

def plot_series(X,
                title='',
                xlabel='x',
                ylabel='y',
                series_labels=None,
                legend=False,
                cmap='Set1',
                figsize=None,
                alpha=0.05):
    """ Plots a time series with confidence intervals

    Arguments:

        X: `ndarray((nsamples, nsteps, nseries))`.
        title: `str`.
        xlabel: `str`.
        ylabel: `str`.
        series_labels: `list`. Labels for each series
        legend: `bool`. Whether to include legend
        cmap: `str`. Which `matplotlib` colormap to use
        figsize: `(width, height)`.
        alpha: `0 <= float <= 1`. Significance threshold (for confidence intervals)

    Returns:

        `matplotlib.pyplot.figure`

    """
    if figsize is None:
        figsize = (6, 2.5)

    Z = ss.norm.ppf(1-(alpha/2))

    nsamples, nsteps, nseries = X.shape
    Xm = np.mean(X, 0)
    Xs = np.std(X, 0)/np.sqrt(nsamples)
    lci = Xm-Z*Xs
    uci = Xm+Z*Xs

    if series_labels is None: 
        series_labels = ['Series %s' %i for i in range(nseries)]

    colors = plt.get_cmap(cmap)
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(nseries):
        ax.fill_between(x=np.arange(nsteps), 
                        y1=lci[:,i], 
                        y2=uci[:,i], 
                        alpha=0.6, 
                        facecolor=colors(i))
        ax.plot(np.arange(nsteps), 
                Xm[:,i], 
                c=colors(i),
                label=series_labels[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if legend: 
        plt.legend()

    return fig


