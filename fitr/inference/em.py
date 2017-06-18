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
from scipy.optimize import brute
from scipy.stats import multivariate_normal as mvn

from .modelfitresult import ModelFitResult

from ..utils import trans_UC
from ..metrics import BIC, AIC, LME

class EM(object):
    """
    Expectation-Maximization with the Laplace Approximation [Huys2011]_, [HuysEMCode]_.

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
    prior : scipy.stats distribution
        The prior distribution over parameter estimates. Here this is fixed to a multivariate normal.
    mu : ndarray(shape=nparams)
        The prior mean over parameters
    cov : ndarray(shape=(nparams,nparams))
        The covariance matrix for prior over parameter estimates

    Methods
    -------
    fit(data, n_iterations=1000, c_limit=1, opt_algorithm='BFGS', diag=False, verbose=True)
        Run the model-fitting algorithm
    logposterior(x, states, actions, rewards)
        Computes the log-posterior probability
    group_level_estimate(param_est, hess_inv)
        Updates the hyperparameters of the group-level prior
    """
    def __init__(self, loglik_func, params, name='EMModel'):
        self.name = name
        self.loglik_func = loglik_func
        self.params = params

        # Extract properties of parameters that will be used in later functions
        self.nparams = len(params)
        self.param_rng = []
        for i in range(self.nparams):
            self.param_rng.append(params[i].rng)


        # Initialize prior
        self.prior = mvn
        self.mu  = np.zeros(self.nparams)
        self.cov = 0.1*np.eye(self.nparams)

    def fit(self, data, n_iterations=1000, c_limit=1e-3, opt_algorithm='L-BFGS-B', init_grid=False, grid_reinit=True, n_grid_points=5, n_reinit=1, dofull=True, early_stopping=True, verbose=True):
        """
        Performs maximum a posteriori estimation of subject-level parameters

        Parameters
        ----------
        data : dict
            Dictionary of data from all subjects.
        n_iterations : int
            Maximum number of iterations to allow.
        c_limit : float
            Threshold at which convergence is determined
        opt_algorithm : {'BFGS', 'L-BFGS-B'}
            Algorithm to use for optimization
        init_grid : bool
            Whether to initialize the optimizer using brute force grid search. If False, will sample from normal distribution with mean 0 and standard deviation 1.
        grid_reinit : bool
            If optimization does not converge, whether to reinitialize with values from grid search
        n_grid_points : int
            Number of points along each axis to evaluate during grid-search initialization (only meaningful if init_grid is True).
        n_reinit : int
            Number of times to reinitialize the optimizer if not converged
        dofull : bool
            Whether update of the full covariance matrix of the prior should be done. If False, the covariance matrix is limited to one in which the off-diagonal elements are set to zero.
        early_stopping : bool
            Whether to stop the EM procedure if the log-model-evidence begins decreasing (thereby reverting to the last iteration's results).
        verbose : bool
            Whether to print progress of model fitting

        Returns
        -------
        ModelFitResult
            Representation of the model fitting results
        """

        # Instantiate the results object
        nsubjects = len(data)
        results = ModelFitResult(method='Expectation-Maximization',
                                 nsubjects=nsubjects,
                                 nparams=self.nparams,
                                 name=self.name)
        results.set_paramnames(params=self.params)


        if init_grid is True:
            init_method = 'Grid Search'
        else:
            init_method = 'Random Initialization'
        print('=============================================\n' +
              '     MODEL: ' + self.name + '\n' +
              '     METHOD: Expectation-Maximization\n' +
              '     INITIALIZATION: ' + init_method + '\n' +
              '     N-RESTARTS: ' + str(n_reinit) + '\n' +
              '     GRID REINITIALIZATION: ' + str(grid_reinit) + '\n' +
              '     MAX EM ITERATIONS: ' + str(n_iterations) + '\n' +
              '     EARLY STOPPING: ' + str(early_stopping) + '\n' +
              '     CONVERGENCE LIMIT: ' + str(c_limit) + '\n' +
              '     OPTIMIZATION ALGORITHM: ' + opt_algorithm + '\n' +
              '     VERBOSE: ' + str(verbose) + '\n' +
              '=============================================\n')

        convergence = False
        opt_iter = 1
        sum_nlogpost = np.sum(results.nlogpost)
        results_old = None # Keep placeholder for old results for early stopping
        while convergence == False and opt_iter < n_iterations:
            for i in range(nsubjects):
                if verbose is True:
                    print('ITERATION: '          + str(opt_iter) +
                          ' | [E-STEP] SUBJECT: ' + str(i+1) +
                          ' | POSTERIOR LOG-LIKELIHOOD: ' + str(-np.round(np.sum(results.nlogpost), 3)))

                # Extract subject-level data
                S = data[i]['S']
                A = data[i]['A']
                R = data[i]['R']

                # Construct the subject's negative log-posterior function
                _nlogpost = lambda x: -self.logposterior(x=x, states=S, actions=A, rewards=R)

                # Run optimization until convergence succeeds
                exitflag = False
                n_converge_fails = 0
                while exitflag is False:
                    if init_grid is True:
                        x0 = self.initialize_opt(fn=_nlogpost,
                                                 grid=True,
                                                 Ns=n_grid_points)
                    else:
                        if grid_reinit is True and n_converge_fails > 0:
                            x0 = self.initialize_opt(fn=_nlogpost,
                                                     grid=True,
                                                     Ns=n_grid_points)
                        else:
                            x0 = self.initialize_opt()

                    # Optimize
                    res = minimize(_nlogpost, x0, method=opt_algorithm)

                    if res.success is False and n_converge_fails < n_reinit:
                        n_converge_fails += 1

                        if verbose is True:
                            print('     FAIL ' + str(n_converge_fails) + ': ' +
                                    str(res.message))

                    elif res.success is True:
                        if verbose is True:

                            print('     SUCCESS: ' + str(res.message))
                        exitflag = res.success
                        update_params = True
                    elif n_converge_fails >= n_reinit:
                        if verbose is True:
                            print('     FAIL ' + str(n_converge_fails) + ': ' +
                                    str(res.message))
                            print('             Using best non-convergent values.')

                        exitflag = True
                        update_params = True


                if update_params == True:
                    # Update subject level data if optimization converged
                    results.nlogpost[i] = res.fun
                    results.params[i,:] = res.x

                    if opt_algorithm=='L-BFGS-B':
                        results.hess[:,:,i] = np.linalg.pinv(res.hess_inv.todense())
                        results.hess_inv[:,:,i] = res.hess_inv.todense()
                    elif opt_algorithm=='BFGS':
                        results.hess[:,:,i] = np.linalg.pinv(res.hess_inv)
                        results.hess_inv[:,:,i] = res.hess_inv

                    results.nloglik[i]  = -self.loglik_func(params=trans_UC(res.x, rng=self.param_rng), states=S, actions=A, rewards=R)
                    results.LME[i] = LME(-res.fun, self.nparams, results.hess[:,:,i])
                    results.AIC[i] = AIC(self.nparams, -results.nloglik[i])
                    results.BIC[i] = BIC(-results.nloglik[i], self.nparams, len(data[0]['A']))

            # If the group level posterior probability has converged, stop
            if np.abs(np.sum(results.nlogpost) - sum_nlogpost) < c_limit:
                convergence = True

                # Add total LME, BIC, and AIC to results list
                results.ts_LME.append(np.sum(results.LME))
                results.ts_nLL.append(np.sum(results.nloglik))
                results.ts_BIC.append(np.sum(results.BIC))
                results.ts_AIC.append(np.sum(results.AIC))
            elif opt_iter > 1 and np.sum(results.LME) < results_old.ts_LME[-1] and early_stopping is True:
                # If the log-model-evidence is DECREASING, then stop
                #   and return the results from the last iteration
                convergence = True

                results = results_old

            else:
                # Update the running model log-posterior probability
                sum_nlogpost = np.sum(results.nlogpost)

                # Add total LME, BIC, and AIC to results list
                results.ts_LME.append(np.sum(results.LME))
                results.ts_nLL.append(np.sum(results.nloglik))
                results.ts_BIC.append(np.sum(results.BIC))
                results.ts_AIC.append(np.sum(results.AIC))

                # Optimize group level hyperparameters
                if verbose is True:
                    print('\nITERATION: ' + str(opt_iter) + ' | [M-STEP]\n')

                self.group_level_estimate(param_est=results.params,
                                          hess_inv=results.hess_inv,
                                          dofull=dofull,
                                          verbose=verbose)

                results_old = results
                opt_iter += 1

        # Transform data to constrained space
        for i in range(nsubjects):
            results.params[i,:] = trans_UC(results.params[i,:], self.param_rng)

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

        lp = self.loglik_func(params=trans_UC(x, rng=self.param_rng), states=states, actions=actions, rewards=rewards) + self.prior.logpdf(x, mean=self.mu, cov=self.cov)

        return lp

    def group_level_estimate(self, param_est, hess_inv, dofull, verbose=True):
        """
        Updates the group-level hyperparameters

        Parameters
        ----------
        param_est : ndarray(shape=(nsubjects, nparams))
            Current parameter estimates for each subject
        hess_inv : ndarray(shape=(nparams, nparams, nsubjects))
            Inverse Hessian matrix estimate for each subject from the iteration with highest log-posterior probability
        dofull : bool
            Whether update of the full covariance matrix of the prior should be done. If False, the covariance matrix is limited to one in which the off-diagonal elements are set to zero.
        verbose : bool
            Controls degree to which results are printed

        """
        nsubjects = np.shape(param_est)[0]

        converged = False
        while converged is False:
            mu_old = self.mu
            self.mu = np.mean(param_est, axis=0)
            #self.mu = np.mean(param_est, axis=0)
            #self.cov = np.zeros([self.nparams, self.nparams])
            cov_new = np.zeros([self.nparams]*2)
            for i in range(nsubjects):
                cov_new = cov_new + (np.outer(param_est[i,:], param_est[i,:]) +  hess_inv[:, :, i]) - np.outer(self.mu, self.mu)
            cov_new = cov_new/(nsubjects-1)

            if np.linalg.det(cov_new) < 0:
                print('     Negative determinant: Prior covariance not updated')
            else:
                self.cov = cov_new

            if dofull is False:
                self.cov = self.cov * np.eye(self.nparams)

            if np.linalg.norm(self.mu-mu_old) < 1e6:
                converged = True
                if verbose is True:
                    print( '     M-STEP CONVERGED \n')

    def initialize_opt(self, fn=None, grid=False, Ns=None):
        """
        Returns initial values for the optimization

        Parameters
        ----------
        fn : function
            Function over which grid search takes place
        grid : bool
            Whether to return initialization values from grid search
        Ns : int
            Number of points per axis over which to evaluate during grid search

        Returns
        -------
        x0 : ndarray
            1 X N vector of initial values for each parameter

        """
        if grid is False:
            # Generate initial values
            x0 = np.random.normal(loc=0, scale=1, size=self.nparams)
        else:
            param_ranges = ()
            for k in range(self.nparams):
                param_ranges = param_ranges + ((-3, 3),)
            res_brute = brute(fn,
                              param_ranges,
                              Ns=Ns,
                              full_output=True)
            if res_brute[0] is not None:
                x0 = res_brute[0]

            else:
                print('     Grid Initialization returned None.')
                x0 = np.random.normal(loc=0, scale=1, size=self.nparams)

        return x0
