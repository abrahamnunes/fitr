# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import matrix_rank
from joblib import Parallel, delayed
from scipy.stats import multivariate_normal as _N
from scipy.optimize import minimize
from fitr.metrics import lme
from fitr.metrics import bic
from fitr.inference import OptimizationResult

def l_bfgs_b(f,
             i,
             data,
             nparams,
             minstarts=2,
             maxstarts=10,
             init_sd=2):
    """ Minimizes the negative log-probability of data with respect to some parameters under function `f` using the L-BFGS-B algorithm.

    This function is specified for use with parallel CPU resources.

    Arguments:

        f: Log likelihood function
        i: `int`. Subject being optimized (slices first dimension of `data`)
        data: Object subscriptable along first dimension to indicate subject being optimized
        nparams: `int`. Number of parameters in the model
        minstarts: `int`. Minimum number of restarts with new initial values
        maxstarts: `int`. Maximum number of restarts with new initial values
        init_sd: Standard deviation for Gaussian initial values

    Returns:

        i: `int`. Subject being optimized (slices first dimension of `data`)
        xmin: `ndarray((nparams,))`. Parameter values at optimum
        fmin: Scalar objective function value at optimum
        fevals: `int`. Number of function evaluations
        niters: `int`. Number of iterations
        lme_: Scalar log-model evidence at optimum
        bic_: Scalar Bayesian Information Criterion at optimum
        hess_inv: `ndarray((nparams, nparams))`. Inv at optimum
    """
    nlog_prob = lambda x: -f(x, data[i])
    fmin    = np.inf
    fevals  = 0
    niters  = 0
    nstarts = 0
    done    = False
    succeeded = False
    while not done:
        xinit = np.random.normal(0, init_sd, size=nparams)
        res = minimize(nlog_prob, xinit, method='L-BFGS-B')

        nstarts += 1
        fevals  += res.nfev
        niters  += res.nit

        # Convergence test
        if nstarts < maxstarts:
            if res.success is True and res.fun < fmin:
                fmin = res.fun
                xmin = res.x
                hess_inv = res.hess_inv.todense()
                lme_ = lme(fmin, nparams, hess_inv)
                bic_ = bic(fmin, nparams, data[i].shape[1])
                succeeded = True
        else:
            done = True
            print('Subject %s Fit | %s Starts | lp_= %s' %(i, nstarts, fmin))
    if succeeded is False:
        raise(ValueError('Failed to converge'))

    return i, xmin, fmin, fevals, niters, lme_, bic_, hess_inv

def mlepar(f,
           data,
           nparams,
           minstarts=2,
           maxstarts=10,
           init_sd=2,
           njobs=-1):
    """ Computes maximum likelihood estimates using parallel CPU resources.

    Wraps over the `fitr.optimization.mle_parallel.mle` function.

    Arguments:

        f: Likelihood function
        data: A subscriptable object whose first dimension indexes subjects
        optimizer: Optimization function (currently only `l_bfgs_b` supported)
        nparams: `int` number of parameters to be estimated
        minstarts: `int`. Minimum number of restarts with new initial values
        maxstarts: `int`. Maximum number of restarts with new initial values
        init_sd: Standard deviation for Gaussian initial values

    Returns:

        `fitr.inference.OptimizationResult`
    """
    nsubjects = len(data)
    plist = [[f, i, data, nparams, minstarts, maxstarts, init_sd] for i in range(nsubjects)]
    y = Parallel(n_jobs=njobs)(delayed(l_bfgs_b)(z[0],z[1],z[2],z[3],z[4],z[5],z[6]) for z in plist)
    res = OptimizationResult(nsubjects, nparams)
    for i, item in enumerate(y):
        sid = item[0]
        res.subject_id[sid] = sid
        res.xmin[sid,:]= item[1]
        res.fmin[sid]=item[2]
        res.fevals[sid] = item[3]
        res.niters[sid] = item[4]
        res.lme[sid]=item[5]
        res.bic[sid] = item[6]
        res.err[sid,:]=np.sqrt(np.diag(item[7]))
        res.hess_inv[sid,:,:]=item[7]
    return res
