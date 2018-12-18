# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gammaln
from fitr.utils import make_onehot
from fitr.stats.distance import distance
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

y = np.random.randint(3, size=100)
y_ = np.random.randint(5, size=100)




def ami(ytrue, ypred, inputs='labels'):
    """ Adjusted mutual information. This is the mutual information between two clusterings adjusted for chance.

    Arguments:

        ytrue: `ndarray((nsamples,))`. True labels. If `inputs='onehot'`, this is `ndarray((nsamples, ntrueclasses))`
        ypred: `ndarray((nsamples,))`. Predicted labels. If `inputs='onehot'`, this is `ndarray((nsamples, npredclasses))`
        inputs: `{'labels', 'onehot'}`. Whether the input labels are onehot encoded or integer/string labeled.

    Returns:

        `float`

    ### TODO:

        - Make this faster

    """
    if inputs == 'labels':
        ytrue, _ = make_onehot(ytrue)
        ypred, _ = make_onehot(ypred)

    nsamples, ntrue = ytrue.shape
    nsamples, npred = ypred.shape

    # Compute joint distribution
    N = ytrue.T@ypred
    pjoint = N/nsamples

    # Compute marginals
    a = np.sum(N, 1)
    b = np.sum(N, 0)
    ptrue = a/nsamples
    ppred = b/nsamples

    # Compute entropies
    Htrue = -ptrue@np.ma.log(ptrue)
    Hpred = -ppred@np.ma.log(ppred)

    # Compute mutual information
    MI = np.sum(pjoint*np.ma.log(pjoint/np.outer(ptrue, ppred)))

    # Compute expected mutual information
    EMI = 0
    Nsgln = gammaln(nsamples + 1)

    for i in range(ntrue):
        for j in range(npred):
            n = np.arange(int(np.maximum(1, a[i] + b[j] - nsamples)), int(np.minimum(a[i], b[j])))
            for _, nij in enumerate(n):
                A = (nij/nsamples)*np.ma.log((nsamples*nij)/(a[i]*b[j]))
                B = gammaln(a[i] + 1) + gammaln(b[j] + 1) + gammaln(nsamples-a[i]+1) + gammaln(nsamples-b[j] + 1)
                C = Nsgln + gammaln(nij + 1) + gammaln(a[i]-nij +1) + gammaln(b[j]-nij+1) + gammaln(nsamples - a[i] - b[j] + nij + 1)
                D = B - C
                EMI += A * np.exp(D)

    # Compute adjusted mutual information
    AMI = (MI - EMI)/(np.maximum(Htrue, Hpred) - EMI)
    return AMI

def homogeneity(ytrue, ypred, inputs='labels'):
    """ Homogeneity score

    Arguments:

        ytrue: `ndarray((nsamples,))`. True labels. If `inputs='onehot'`, this is `ndarray((nsamples, ntrueclasses))`
        ypred: `ndarray((nsamples,))`. Predicted labels. If `inputs='onehot'`, this is `ndarray((nsamples, npredclasses))`
        inputs: `{'labels', 'onehot'}`. Whether the input labels are onehot encoded or integer/string labeled.

    Returns:

        `float`

    ### TODO:

        - Make this faster

    """
    if inputs == 'labels':
        ytrue, _ = make_onehot(ytrue)
        ypred, _ = make_onehot(ypred)

    nsamples, ntrue = ytrue.shape
    nsamples, npred = ypred.shape

    N = ytrue.T@ypred
    ntrue = np.sum(N, 1)
    npred = np.sum(N, 0)

    Hck = -np.sum((N/nsamples)*np.ma.log(N/npred))
    Hc  = -np.sum((ntrue/nsamples)*np.ma.log(ntrue/nsamples))

    return 1 - (Hck/Hc)

def completeness(ytrue, ypred, inputs='labels'):
    """ Completeness score

    Arguments:

        ytrue: `ndarray((nsamples,))`. True labels. If `inputs='onehot'`, this is `ndarray((nsamples, ntrueclasses))`
        ypred: `ndarray((nsamples,))`. Predicted labels. If `inputs='onehot'`, this is `ndarray((nsamples, npredclasses))`
        inputs: `{'labels', 'onehot'}`. Whether the input labels are onehot encoded or integer/string labeled.

    Returns:

        `float`

    ### TODO:

        - Make this faster

    """
    if inputs == 'labels':
        ytrue, _ = make_onehot(ytrue)
        ypred, _ = make_onehot(ypred)

    nsamples, ntrue = ytrue.shape
    nsamples, npred = ypred.shape

    N = ytrue.T@ypred
    ntrue = np.sum(N, 1)
    npred = np.sum(N, 0)

    Hkc = -np.sum((N.T/nsamples)*np.ma.log(N.T/ntrue))
    Hk  = -np.sum((npred/nsamples)*np.ma.log(npred/nsamples))

    return 1 - (Hkc/Hk)


def v_measure(ytrue, ypred, inputs='labels'):
    """ V-Measure score

    Arguments:

        ytrue: `ndarray((nsamples,))`. True labels. If `inputs='onehot'`, this is `ndarray((nsamples, ntrueclasses))`
        ypred: `ndarray((nsamples,))`. Predicted labels. If `inputs='onehot'`, this is `ndarray((nsamples, npredclasses))`
        inputs: `{'labels', 'onehot'}`. Whether the input labels are onehot encoded or integer/string labeled.

    Returns:

        `float`

    ### TODO:

        - Make this faster

    """
    h = homogeneity(ytrue, ypred, inputs=inputs)
    c = completeness(ytrue, ypred, inputs=inputs)
    return 2*((h*c)/(h+c))

def silhouette(D, y, inputs='labels', return_samples=False):
    """ Computes the silhouette statistic for clustering

    Arguments:

        D: `ndarray((nsamples, nsamples))`. Distance matrix.
        y: `ndarray((nsamples,))`. Labels. If `labels = 'onehot'`, then this will be `ndarray((nsamples,nclusters))`.
        inputs: `{'labels', 'onehot'}`.
        return_samples: `bool`. Whether to return the silhouette scores for individual samples

    Returns:

        `float`

    """
    if inputs == 'onehot':
        y = np.argmax(y, 1)

    clusters = np.unique(y)
    nclusters = clusters.size
    nsamples = y.size

    d = np.empty((nsamples, nclusters))
    a = np.empty(nsamples)
    b = np.empty(nsamples)
    s = np.empty(nsamples)
    for i in range(nsamples):
        for j in range(nclusters):
            idx = np.equal(y, j)
            idx[i] = False
            d[i,j] = np.mean(D[i,idx])

        ci = np.equal(clusters, y[i])
        a[i] = d[i, ci]
        b[i] = np.min(d[i, np.logical_not(ci)])
        s[i] = (b[i] - a[i])/np.maximum(a[i], b[i])

    if return_samples:
        return s
    else:
        return np.mean(s)
