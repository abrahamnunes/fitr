# -*- coding: utf-8 -*-
import numpy as np

class OptimizationResult(object):
    """ Container for the results of an optimization run on a generative model of behavioural data

    Arguments:

        subject_id: `ndarray((nsubjects,))` or `None` (default). Integer ids for subjects
        xmin: `ndarray((nsubjects,nparams))` or `None` (default). Parameters that minimize objective function
        fmin: `ndarray((nsubjects,))` or `None` (default). Value of objective function at minimum
        fevals: `ndarray((nsubjects,))` or `None` (default). Number of function evaluations required to minimize objective function
        niters: `ndarray((nsubjects,))` or `None` (default). Number of iterations required to minimize objective function
        lme: `ndarray((nsubjects,))` or `None` (default). Log model evidence
        bic: `ndarray((nsubjects,))` or `None` (default). Bayesian Information Criterion
        hess_inv: `ndarray((nsubjects,nparams,nparams))` or `None` (default). Inverse Hessian at the optimum.
        err: `ndarray((nsubjects,nparams))` or `None` (default). Error of estimates at optimum.
    """

    def __init__(self,
                 nsubjects=None,
                 nparams=None,
                 subject_id=None,
                 xmin=None,
                 fmin=None,
                 fevals=None,
                 niters=None,
                 lme=None,
                 bic=None,
                 hess_inv=None,
                 err=None):
        self.nsubjects = nsubjects
        self.nparams = nparams

        if subject_id is None: self.subject_id = np.empty(nsubjects)
        else: self.subject_id = subject_id

        if xmin is None: self.xmin = np.empty((nsubjects, nparams))
        else: self.xmin = xmin

        if fmin is None: self.fmin = np.empty(nsubjects)
        else: self.fmin = fmin

        if fevals is None: self.fevals = np.empty(nsubjects)
        else: self.fevals = fevals

        if niters is None: self.niters = np.empty(nsubjects)
        else: self.niters = niters

        if lme is None: self.lme = np.empty(nsubjects)
        else: self.lme = lme

        if bic is None: self.bic = np.empty(nsubjects)
        else: self.bic = bic

        if hess_inv is None: self.hess_inv = np.empty((nsubjects, nparams, nparams))
        else: self.hess_inv = hess_inv

        if err is None: self.err = np.empty((nsubjects, nparams))
        else: self.err = err

    def transform_xmin(self, transforms, inplace=False):
        """ Rescales the parameter estimates.

        Arguments:

            transforms: `list`. Transformation functions where `len(transforms) == self.xmin.shape[1]`
            inplace: `bool`. Whether to change the values in `self.xmin`. Default is `False`, which returns an `ndarray((nsubjects, nparams))` of the transformed parameters.

        Returns:

            `ndarray((nsubjects, nparams))` of the transformed parameters if `inplace=False`
        """
        X = np.empty(self.xmin.shape)
        for i, f_trans in enumerate(transforms):
            X[:,i] = f_trans(self.xmin[:,i])

        if inplace: self.xmin = X
        else: return X
