import numpy as np
import pandas as pd

#===============================================================================
#   SUCRA
#===============================================================================

def SUCRA(X, order='ascending'):
    """ Computes the surface under the cumulative ranking curve, along with cumulative ranking probabilities and expected rank.

    Arguments:

        X: `ndarray((nsamples, ncomparators))`. Matrix of integer valued ranks
        order: `str`. Set to `order='ascending'` if 0=lowest and `ncomparators-1`=highest, and vice versa

    Returns:

        sucra: `ndarray(ncomparators)`. Surface under the cumulative ranking curve
        expected_rank: `ndarray(ncomparators)`. Expected rank of effect for each covariate
        P: `ndarray((ncomparators, ncomparators))`. Probability of each rank (columns) for each comparator (rows)
        F: `ndarray((ncomparators, ncomparators))`. Cumulative probability (that each comparator has rank j or better)
    """
    _, ncomparators  = X.shape
    unique_ranks = np.unique(X)
    P = np.empty((ncomparators, ncomparators))

    for i in range(ncomparators):
        rcov = X[:,i]
        for j in range(ncomparators):
            P[i,j] = np.mean(np.equal(rcov, j))

    if order == 'ascending':
        P = np.flip(P, axis=1)

    F = np.cumsum(P, axis=1)
    sucra = (1/(ncomparators-1))*np.sum(F[:,:-1], axis=1)
    expected_rank = ncomparators - np.sum(F[:,:-1], axis=1)
    return sucra, expected_rank, P, F

#===============================================================================
#   HETEROGENEITY
#===============================================================================

class HeterogeneityResult(object):
    """

    Attributes:

        nstudies: `int`. Number of studies
        df: `float`. Degrees of freedom
        es: `ndarray(nstudies)`. Effect sizes
        summary_es: `float`. Summary effect size from the analysis
        weights: `ndarray(nstudies)`. Study weights
        Q: `float`. Cochran's Q
        ev: `float`. Excess variance $Q - df$
        T2: `float`. Estimate of $\\tau^2$
        i2: `float`. Estimate of $i^2$

    """
    def __init__(self):
        self.nstudies = None
        self.df = None
        self.es = None
        self.summary_es = None
        self.weights = None
        self.Q = None
        self.ev = None
        self.T2 = None
        self.i2 = None

def cochrans_q(es, summary_es, weights):
    """ Computes Cochran's Q statistic for heterogeneity

    Arguments:

        es: `ndarray(nstudies)`
        summary_es: `ndarray(nstudies)`
        weights: `ndarray(nstudies)`

    Returns:

        `float`
    """
    es = es.flatten()
    weights = weights.flatten()
    squared_deviations = np.square(es-summary_es)
    return np.einsum('i,j->', weights, squared_deviations)

def excess_variation(Q, nstudies):
    """ Computes excess variation, given that Cochran's Q is the observed weighted sum of squares and degrees of freedom are the expected weighted sum of squares.

    The excess variation is what can be attributed to differences in the true effects between studies.

    Arguments:

        Q: `float`. Cochran's Q
        nstudies: `int`. Number of studies

    Returns:

        `float`
    """
    return Q - (nstudies - 1)

def tau2(ev, weights):
    """ Computes an estimate of the variance of the true effect sizes

    $$
    T^2 = \\frac{Q-df}{K} \\approx \\tau^2
    $$

    $$
    K = \\sum_i w_i = \\frac{\\sum_i w_i^2}{\\sum_i w_i}
    $$

    Arguments:

        ev: `float`. Excess variation
        weights: `ndarray(nstudies)`. Weights for each study

    Returns:

        `float`

    """
    K = np.sum(weights) - (np.sum(weights**2)/np.sum(weights))
    return np.maximum(ev/K, 0.)

def i2(Q, df):
    """ Computes an estimate of the degree to which observed variance reflects true effect size differences

    $$
    i^2 = \\frac{Q-df}{Q}
    $$

    Arguments:

        Q: `float`. Cochran's Q
        df: `float`. Degrees of freedom `df = nstudies - 1`

    Returns:

        `float`

    """
    return np.maximum((Q-df)/Q, 0.)


def heterogeneity(es, summary_es, weights):
    """ Given data from a meta-analysis, returns statistics related to heterogeneity

    Arguments:

        es: `ndarray(nstudies)`. Effect sizes for each study
        summary_es: `float`. Summary effect size
        weights: `ndarray(nstudies)`. Weights for each study

    Returns:

       `HeterogeneityResult` object

    """
    res = HeterogeneityResult()
    res.nstudies = es.size
    res.df = res.nstudies - 1
    res.es = es.flatten()
    res.summary_es = summary_es
    res.weights = weights.flatten()
    res.Q = cochrans_q(res.es, res.summary_es, res.weights)
    res.ev = res.Q - res.df
    res.T2 = tau2(res.ev, res.weights)
    res.i2 = i2(res.Q, res.df)
    return res
