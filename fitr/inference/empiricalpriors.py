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

import numpy as np
from scipy.optimize import minimize

from .modelfitresult import OptimizationFitResult

from ..utils import trans_UC
from ..criticism.model_evaluation import BIC, AIC, LME

class EmpiricalPriors(object):
    """
    Inference procedure with empirical priors

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
    logposterior(x, states, actions, rewards)
        Computes the log-poseterior probability
    __printfitstart(self, n_iterations, algorithm, verbose)
        (Private) function to print optimization info to console
    __printupdate(self, opt_iter, subject_i, posterior_ll, verbose)
        (Private) function to print iteration info to console
    """
    def __init__(self, loglik_func, params, name='EmpiricalPriorsModel'):
        self.name = name
        self.loglik_func = loglik_func
        self.params = params

        # Extract properties of parameters that will be used in later functions
        self.nparams = len(params)
        self.param_rng = []
        for i in range(self.nparams):
            self.param_rng.append(params[i].rng)

    def fit(self, data, n_iterations=1000, c_limit=1e-3, opt_algorithm='L-BFGS-B', verbose=True):
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
        results = OptimizationFitResult(method='Empirical Priors',
                                        nsubjects=nsubjects,
                                        nparams=self.nparams,
                                        name=self.name)
        results.set_paramnames(params=self.params)

        # Print initial message to console
        self.__printfitstart(n_iterations=n_iterations,
                             algorithm=opt_algorithm,
                             verbose=verbose)

        convergence = False
        opt_iter = 1
        sum_nlogpost = 0 # Monitor total neg-log-posterior for convergence
        while convergence is False and opt_iter < n_iterations:
            for i in range(nsubjects):
                # Print update message to console
                self.__printupdate(curr_iter=opt_iter,
                                   subject_i=i,
                                   _lp=-np.round(np.sum(results.nlogpost), 3),
                                   verbose=verbose)

                # Construct subjects negative log-posterior function
                def _nlogpost(x):
                    _lp = -self.loglik_func(params=x,
                                            states=data[i]['S'],
                                            actions=data[i]['A'],
                                            rewards=data[i]['R'])
                    return _lp


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
                        for k in range(self.nparams):
                            x0[k] = self.params[k].sample()
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

                    trans_params = trans_UC(res.x, rng=self.param_rng)
                    results.nloglik[i] = -self.loglik_func(params=trans_params,
                                                           states=data[i]['S'],
                                                           actions=data[i]['A'],
                                                           rewards=data[i]['R'])
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

    def logposterior(self, x, states, actions, rewards):
        """
        Represents the log-posterior probability function

        Parameters
        ----------
        x : ndarray(nparams)
            Array of parameters for single subject
        states : ndarray
            Array of states encountered by subject. Number of rows should reflect number of trials. If the task is a multi-step per trial task, then the number of columns should reflect the number of steps, unless a custom likelihood function is used which does not require this.
        actions: ndarray
            Array of actions taken by subject. Number of rows should reflect number of trials. If the task is a multi-step per trial task, then the number of columns should reflect the number of steps, unless a custom likelihood function is used which does not require this.
        rewards : ndarray
            Array of rewards received by the subject. Number of rows should reflect number of trials. If there are multiple steps at which rewards are received, they should be stored in different columns, unless a custom likelihood funciton is used which does not require this.

        Returns
        -------
        float
            Log-posterior probability
        """

        lp = self.loglik_func(params=x, states=states, actions=actions, rewards=rewards)

        for i in range(self.nparams):
            lp = lp + self.params[i].dist.logpdf(x[i])

        return lp

    def __printfitstart(self, n_iterations, algorithm, verbose):
        """
        Prints information in console banner when fitting starts

        Parameters
        ----------
        n_iterations : int
            Maximum number of iterations to allow.
        algorithm : {'BFGS', 'L-BFGS-B'}
            Algorithm to use for optimization
        verbose : bool
            Whether to print progress of model fitting
        """
        print('=============================================\n' +
              '     MODEL: ' + self.name + '\n' +
              '     METHOD: Empirical Priors\n' +
              '     ITERATIONS: ' + str(n_iterations) + '\n' +
              '     OPTIMIZATION ALGORITHM: ' + algorithm + '\n' +
              '     VERBOSE: ' + str(verbose) + '\n' +
              '=============================================\n')

    @classmethod
    def __printupdate(self, curr_iter, subject_i, _lp, verbose):
        """
        Prints update on iteration fit

        Parameters
        ----------
        opt_iter : int > 0
            Current iteration of optimization
        subject_i : int >= 0
            Current subject index
        _lp : float
            Current posterior log-likelihood
        verbose : bool
            Whether to print
        """
        if verbose is True:
            print('ITERATION: '          + str(curr_iter) +
                  ' | SUBJECT: ' + str(subject_i+1) +
                  ' | POSTERIOR LOG-LIKELIHOOD: ' + str(_lp))
