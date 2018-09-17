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
             maxstarts_without_improvement=3,
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
        maxstarts_without_improvement: `int`. Maximum number of restarts without improvement in objective function value
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
    nstarts_without_improvement = 0
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

            if res.fun >= fmin:
                nstarts_without_improvement += 1
                if nstarts_without_improvement >= maxstarts_without_improvement:
                    done = True
                    print('Subject %s Fit | %s Starts | N Fx Evals %s | lp_= %s' %(i, nstarts, fevals, fmin))

        else:
            done = True
            print('Subject %s Fit | %s Starts | N Fx Evals %s | lp_= %s' %(i, nstarts, fevals, fmin))
    if succeeded is False:
        print('Subject %s failed to converge after %s iterations (%s fx evals)' %(i, niters, fevals))
        fmin = np.nan
        xmin = np.array([np.nan]*xinit.size)
        hess_inv = np.array([[np.nan]*xinit.size]*xinit.size)
        lme_ = np.nan
        bic_ = np.nan
        succeeded = False

    return i, xmin, fmin, fevals, niters, lme_, bic_, hess_inv

def second_order_optimizer(f,
                           i,
                           data,
                           nparams,
                           jac,
                           hess,
                           minstarts=2,
                           maxstarts=10,
                           maxstarts_without_improvement=3,
                           init_sd=2,
                           method='trust-exact'):
    """ Minimizes the negative log-probability of data with respect to some parameters under function `f` using the exact .

    This function is specified for use with parallel CPU resources.

    Arguments:

        f: Log likelihood function
        i: `int`. Subject being optimized (slices first dimension of `data`)
        data: Object subscriptable along first dimension to indicate subject being optimized
        nparams: `int`. Number of parameters in the model
        jac: `bool`. Set to `True` if `f` returns a Jacobian as the second element of the returned values
        hess: `bool`. Set to `True` if third output value of `f` is the Hessian matrix
        minstarts: `int`. Minimum number of restarts with new initial values
        maxstarts: `int`. Maximum number of restarts with new initial values
        maxstarts_without_improvement: `int`. Maximum number of restarts without improvement in objective function value
        init_sd: Standard deviation for Gaussian initial values

    Returns:

        i: `int`. Subject being optimized (slices first dimension of `data`)
        xmin: `ndarray((nparams,))`. Parameter values at optimum
        fmin: Scalar objective function value at optimum
        fevals: `int`. Number of function evaluations
        niters: `int`. Number of iterations
        lme_: Scalar log-model evidence at optimum
        bic_: Scalar Bayesian Information Criterion at optimum
        hess: `ndarray((nparams, nparams))`. Inv at optimum
    """
    nlog_prob = lambda x: f(x, data[i])[:-1]
    hessian = lambda x: f(x, data[i])[-1]
    fmin    = np.inf
    fevals  = 0
    niters  = 0
    nstarts = 0
    nstarts_without_improvement = 0
    done    = False
    succeeded = False
    while not done:
        xinit = np.random.normal(0, init_sd, size=nparams)
        res = minimize(nlog_prob,
                       xinit,
                       jac=jac,
                       hess=hessian,
                       method=method)

        nstarts += 1
        fevals  += res.nfev
        niters  += res.nit

        # Convergence test
        if nstarts < maxstarts:
            if res.success is True and res.fun < fmin:
                fmin = res.fun
                xmin = res.x
                hess_inv = np.linalg.pinv(res.hess)
                lme_ = lme(fmin, nparams, hess_inv)
                bic_ = bic(fmin, nparams, data[i].shape[1])
                succeeded = True

            if res.fun >= fmin:
                nstarts_without_improvement += 1
                if nstarts_without_improvement >= maxstarts_without_improvement:
                    done = True
                    print('Subject %s Fit | %s Starts | N Fx Evals %s | lp_= %s' %(i, nstarts, fevals, fmin))
        else:
            done = True
            print('Subject %s Fit | %s Starts | N Fx Evals %s | lp_= %s' %(i, nstarts, fevals, fmin))
    if succeeded is False:
        print('Subject %s failed to converge after %s iterations (%s fx evals)' %(i, niters, fevals))
        fmin = np.nan
        xmin = np.array([np.nan]*xinit.size)
        hess_inv = np.array([[np.nan]*xinit.size]*xinit.size)
        lme_ = np.nan
        bic_ = np.nan
        succeeded = False

    return i, xmin, fmin, fevals, niters, lme_, bic_, hess_inv

def mlepar(f,
           data,
           nparams,
           minstarts=2,
           maxstarts=10,
           maxstarts_without_improvement=3,
           init_sd=2,
           njobs=-1,
           jac=None,
           hess=None,
           method='L-BFGS-B'):
    """ Computes maximum likelihood estimates using parallel CPU resources.

    Wraps over the `fitr.optimization.mle_parallel.mle` function.

    Arguments:

        f: Likelihood function
        data: A subscriptable object whose first dimension indexes subjects
        optimizer: Optimization function (currently only `l_bfgs_b` supported)
        nparams: `int` number of parameters to be estimated
        minstarts: `int`. Minimum number of restarts with new initial values
        maxstarts: `int`. Maximum number of restarts with new initial values
        maxstarts_without_improvement: `int`. Maximum number of restarts without improvement in objective function value
        init_sd: Standard deviation for Gaussian initial values
        jac: `bool`. Set to `True` if `f` returns a Jacobian as the second element of the returned values
        hess: `bool`. Set to `True` if third output value of `f` is the Hessian matrix
        method: `str`. One of the `scipy.optimize` methods.

    Returns:

        `fitr.inference.OptimizationResult`

    Todo:

        - [ ] Raise errors when user selects inappropriate optimization function given values for `jac` and `hess`

    """
    nsubjects = len(data)

    if method == 'L-BFGS-B':
        plist = [[f, i, data, nparams, minstarts, maxstarts, maxstarts_without_improvement, init_sd] for i in range(nsubjects)]
        y = Parallel(n_jobs=njobs)(delayed(l_bfgs_b)(z[0],z[1],z[2],z[3],z[4],z[5],z[6],z[7]) for z in plist)

    elif method in ['trust-exact', 'trust-ncg', 'trust-krylov', 'dogleg']:
        plist = [[f, i, data, nparams, jac, hess, minstarts, maxstarts, maxstarts_without_improvement, init_sd, method] for i in range(nsubjects)]
        y = Parallel(n_jobs=njobs)(delayed(second_order_optimizer)(z[0],z[1],z[2],z[3],z[4],z[5],z[6],z[7],z[8],z[9],z[10]) for z in plist)

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
