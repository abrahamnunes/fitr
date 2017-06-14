# -*- coding: utf-8 -*-
# Fitr. A package for fitting reinforcement learning models to behavioural data
# Copyright (C) 2017 Abraham Nunes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# CONTACT INFO:
#   Abraham Nunes
#    Email: nunes@dal.ca
#
# ============================================================================

import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.stats import multivariate_normal as mvn

from .modelfitresult import ModelFitResult

from ..utils import trans_UC
from ..metrics import BIC, AIC, LME

class MLE(object):
    """
    Maximum Likelihood parameter estimation

    Attributes
    ----------
    name : str
        Name of the model being fit. We suggest using the free parameters.
    loglik_func : function
        The log-likelihood function to be used for model fitting
    params : list
        List of parameters from the rlparams module
    nparams : int
        Number of free parameters in the model
    param_rng : list
        List of strings denoting the parameter ranges (see rlparams module for further details)

    Methods
    -------
    fit(data, n_iterations=1000, opt_algorithm='BFGS')
        Runs model-fitting algorithm
    """
    def __init__(self, loglik_func, params, name='MLEModel'):
        self.name = name
        self.loglik_func = loglik_func
        self.params = params

        # Extract properties of parameters that will be used in later functions
        self.nparams = len(params)
        self.param_rng = []
        for i in range(self.nparams):
            self.param_rng.append(params[i].rng)

    def fit(self, data, n_iterations=1000, c_limit=1e-4, opt_algorithm='L-BFGS-B', verbose=True):
        """
        Runs the maximum a posterior model-fitting with empirical priors.

        Parameters
        ----------
        data : dict
            Dictionary of data from all subjects.
        n_iterations : int
            Maximum number of iterations to allow.
        c_limit : float
            Threshold at which convergence is determined
        opt_algorithm : {'L-BFGS-B'}
            Algorithm to use for optimization. Only works at present with L-BFGS-B.
        verbose : bool
            Whether to print progress of model fitting

        Returns
        -------
        ModelFitResult
            Representation of the model fitting results
        """

        # Instantiate the results object
        nsubjects = len(data)
        results = ModelFitResult(method='Maximum Likelihood',
                                 nsubjects=nsubjects,
                                 nparams=self.nparams,
                                 name=self.name)
        results.set_paramnames(params=self.params)

        print('=============================================\n' +
              '     MODEL: ' + self.name + '\n' +
              '     METHOD: Maximum Likelihood\n' +
              '     ITERATIONS: ' + str(n_iterations) + '\n' +
              '     OPTIMIZATION ALGORITHM: ' + opt_algorithm + '\n' +
              '     VERBOSE: ' + str(verbose) + '\n' +
              '=============================================\n')

        convergence = False
        opt_iter = 1
        sum_nlogpost = 0 # Monitor total neg-log-posterior for convergence
        while convergence == False and opt_iter < n_iterations:
            for i in range(nsubjects):
                if verbose is True:
                    print('ITERATION: '  + str(opt_iter) +
                          ' | SUBJECT: ' + str(i+1))

                # Extract subject-level data
                S = data[i]['S']
                A = data[i]['A']
                R = data[i]['R']

                # Construct the subject's negative log-posterior function
                _nlogpost = lambda x: -self.loglik_func(params=x, states=S, actions=A, rewards=R)

                # Create bounds
                bounds = ()
                for k in range(self.nparams):
                    if self.params[k].rng == 'unit':
                        bounds = bounds + ((0, 1),)
                    elif self.params[k].rng == 'pos':
                        bounds = bounds + ((0,100),)
                    elif self.params[k].rng == 'unc':
                        bounds = bounds + ((-1000, 1000),)

                # Run optimization until convergence succeeds
                exitflag = False
                n_converge_fails = 0
                np.seterr(divide='ignore', invalid='ignore')
                while exitflag is False:
                    # Generate initial values
                    x0 = np.zeros(self.nparams)
                    isfin = False
                    while isfin is False:
                        ranges = []
                        for k in range(self.nparams):
                            ranges.append(self.params[k].rng)
                        rv = np.random.normal(0, 1, size=self.nparams)
                        x0 = trans_UC(rv, rng=ranges)
                        lp = _nlogpost(x0)
                        isfin = np.isfinite(lp)

                    # Optimize
                    if opt_algorithm == 'L-BFGS-B':
                        res = minimize(_nlogpost, x0, bounds=bounds, method=opt_algorithm)
                    elif opt_algorithm == 'BFGS':
                        res = minimize(_nlogpost, x0, method=opt_algorithm)

                    if res.success is False:
                        n_converge_fails += 1
                        if verbose is True:
                            print('     Failed to converge ' +
                                    str(n_converge_fails) + ' times.')

                    exitflag = res.success

                # Update subject level data if logposterior improved
                if res.fun < results.nlogpost[i]:
                    results.nlogpost[i] = res.fun
                    results.params[i,:] = res.x

                    if opt_algorithm=='L-BFGS-B':
                        results.hess[:,:,i] = np.linalg.inv(res.hess_inv.todense())
                        results.hess_inv[:,:,i] = res.hess_inv.todense()
                    elif opt_algorithm=='BFGS':
                        results.hess[:,:,i] = np.linalg.inv(res.hess_inv)
                        results.hess_inv[:,:,i] = res.hess_inv

                    results.nloglik[i]  = res.fun
                    results.LME[i] = LME(-res.fun, self.nparams, results.hess[:,:,i])
                    results.AIC[i] = AIC(self.nparams, -results.nloglik[i])
                    results.BIC[i] = BIC(-results.nloglik[i], self.nparams, len(data[0]['A']))

            # If the group level posterior probability has converged, stop
            if np.abs(np.sum(results.nlogpost) - sum_nlogpost) < c_limit:
                convergence = True
            else:
                # Update the running model log-posterior probability
                sum_nlogpost = np.sum(results.nlogpost)

                # Add total LME, BIC, and AIC to results list
                results.ts_LME.append(np.sum(results.LME))
                results.ts_nLL.append(np.sum(results.nloglik))
                results.ts_BIC.append(np.sum(results.BIC))
                results.ts_AIC.append(np.sum(results.AIC))

                opt_iter += 1

        print('\n MODEL FITTING COMPLETED \n')

        # Generate summary table in results
        results.summary_table()

        return results
