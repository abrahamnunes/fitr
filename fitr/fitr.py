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
import pandas as pd
import pystan
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.stats import multivariate_normal as mvn

from .utils import trans_UC, BIC, AIC, LME

# ==============================================================================
#
#   FITRMODEL
#       The highest level object representing a task model to be fit.
#       It is not necessary to use this object, since one could use the EM
#        object (lower level), but this method should be the simplest for
#        inexperienced users
#
# ==============================================================================

class fitrmodel(object):
    """
    An object representing a model to be fit to behavioural data. This should be viewed as a high level wrapper for multiple potential model fitting algorithms which themselves can be run by using their respective classes.

    Attributes
    ----------
    name : str
        Name of the model. We suggest identifying model based on free parameters.
    loglik_func : function
        The log-likelihood function to be used to fit the data
    params : list
        List of reinforcement learning parameter objects from the rlparams module.
    generative_model : GenerativeModel object
        Object representing a generative model

    Methods
    -------
    fit(data, method='EM', c_limit=0.01)
        Runs the specified model fitting algorithm with the given data.
    """
    def __init__(self, name='Anon Model', loglik_func=None, params=None, generative_model=None):
        self.name = name
        self.loglik_func = loglik_func
        self.generative_model = generative_model
        self.params = params

    def fit(self, data, method='EM', c_limit=0.01, verbose=True):
        """
        Runs model fitting

        Parameters
        ----------
        data : dict
            Behavioural data.
        method : {'EM', 'MLE', 'MAP0', 'EmpiricalPriors', 'MCMC'}
            The inference algorithm to use. Note that the data formats for 'MCMC' compared to the other methods is distinct, and should correspond appropriately to the method being employed
        c_limit : float
            Limit at which convergence of log-posterior probability is determined (only for methods 'EM' and 'EmpiricalPriors')
        verbose : bool
            Controls amount of printed output during model fitting

        Returns
        -------
        fitrfit : object
            Representation of the model fitting results
        """

        if method=='EM':
            m = EM(loglik_func=self.loglik_func,
                   params=self.params,
                   name=self.name)
            results = m.fit(data=data,
                            c_limit=c_limit,
                            verbose=verbose)
        elif method=='MLE':
            m = EM(loglik_func=self.loglik_func,
                   params=self.params,
                   name=self.name)
            results = m.fit(data=data,
                            n_iterations=1,
                            c_limit=c_limit,
                            verbose=verbose)
        elif method=='MAP0':
            m = EM(loglik_func=self.loglik_func,
                   params=self.params,
                   name=self.name)
            results = m.fit(data=data,
                            n_iterations=2,
                            c_limit=c_limit,
                            verbose=verbose)
        elif method=='EmpiricalPriors':
            m = EmpiricalPriors(loglik_func=self.loglik_func,
                                params=self.params,
                                name=self.name)
            results = m.fit(data=data,
                            c_limit=c_limit,
                            verbose=verbose)
        elif method=='MCMC':
            m = MCMC(generative_model=self.generative_model, name=self.name)
            results = m.fit(data=data)

        return results

#===============================================================================
#
#   INFERENCE METHOD OBJECTS
#       - EM: Expectation-Maximization with Laplace Approximation
#       - EmpiricalPriors
#       - MCMC: Markov-Chain Monte-Carlo
#
#===============================================================================

# EXPECTATION-MAXIMIZATION METHOD
class EM(object):
    """
    Expectation-Maximization with the Laplace Approximation

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

    References
    ----------
    [1] Huys, Q. J. M., et al. (2011). Disentangling the roles of approach, activation and valence in instrumental and pavlovian responding. PLoS Computational Biology, 7(4).
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
        fitrfit : object
            Representation of the model fitting results
        """

        # Instantiate the results object
        nsubjects = len(data)
        results = fitrfit(method='Expectation-Maximization',
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

        References
        ----------
        [1] Quentin Huys' `emfit.m` code at https://bitbucket.org/fpetzschner/cpc2016/src
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

# EMPIRICAL PRIORS METHOD
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
        fitrfit
            Representation of the model fitting results
        """

        # Instantiate the results object
        nsubjects = len(data)
        results = fitrfit(method='Empirical Priors',
                          nsubjects=nsubjects,
                          nparams=self.nparams,
                          name=self.name)
        results.set_paramnames(params=self.params)

        print('=============================================\n' +
              '     MODEL: ' + self.name + '\n' +
              '     METHOD: Empirical Priors\n' +
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
                    print('ITERATION: '          + str(opt_iter) +
                          ' | SUBJECT: ' + str(i+1))

                # Extract subject-level data
                S = data[i]['S']
                A = data[i]['A']
                R = data[i]['R']

                # Construct the subject's negative log-posterior function
                _nlogpost = lambda x: -self.logposterior(x=x, states=S, actions=A, rewards=R)


                # Create bounds
                bounds = ()
                for k in range(self.nparams):
                    if self.params[k].rng == 'unit':
                        bounds = bounds + ((0, 1),)
                    elif self.params[k].rng == 'pos':
                        bounds = bounds + ((0,100),)
                    elif self.params[k].rng == 'neg':
                        bounds = bounds + ((-100, 0),)
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

                    results.nloglik[i]  = -self.loglik_func(params=trans_UC(res.x, rng=self.param_rng), states=S, actions=A, rewards=R)
                    results.LME[i] = LME(-res.fun, self.nparams, results.hess[:,:,i])
                    results.AIC[i] = AIC(self.nparams, results.nloglik[i])
                    results.BIC[i] = BIC(results.nloglik[i], self.nparams, len(data[0]['A']))

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

class MCMC(object):
    """
    Uses Markov-Chain Monte-Carlo (via PyStan) to estimate models

    Parameters
    ----------
    name : str
        Name of the model being fit
    generative_model : GenerativeModel object
    """
    def __init__(self, generative_model=None, name='FitrMCMCModel'):
        self.name = name
        self.generative_model = generative_model

    def fit(self, data, chains=4, n_iterations=2000, warmup=None, thin=1, seed=None, init='random', sample_file=None, algorithm='NUTS', control=None, n_jobs=-1, compile_verbose=False, sampling_verbose=False):
        """
        Runs the MCMC Inference procedure with Stan

        Parameters
        ----------
        data : dict
            Subject level data
        chains : int > 0
            Number of chains in sampler
        n_iter : int
            How many iterations each chain should run (includes warmup)
        warmup : int > 0, iter//2 by default
            Number of warmup iterations.
        thin : int > 0
            Period for saving samples
        seed : int or np.random.RandomState, optional
            Positive integer to initialize random number generation
        sample_file : str
            File name specifying where samples for all parameters and other saved quantities will be written. If None, no samples will be written
        algorithm : {'NUTS', 'HMC', 'Fixed_param'}, optional
            Which of Stan's algorithms to implement
        control : dict, optional
            Dictionary of parameters to control sampler's behaviour (see PyStan documentation for details)
        n_jobs : int, optional
            Sample in parallel. If -1, all CPU cores are used. If 1, no parallel computing is used
        compile_verbose : bool
            Whether to print output from model compilation
        sampling_verbose : bool
            Whether to print intermediate output from model sampling

        Returns
        -------
        fitrfit
            Instance containing model fitting results

        References
        ----------
        [1] PyStan API documentation (https://pystan.readthedocs.io)
        """

        print('=============================================\n' +
              '     MODEL: ' + self.name + '\n' +
              '     METHOD: Markov Chain Monte-Carlo\n' +
              '     ITERATIONS: ' + str(n_iterations) + '\n' +
              '     OPTIMIZATION ALGORITHM: ' + algorithm + '\n' +
              '=============================================\n')

        # Instantiate a fitrfit object
        results = fitrfit(name=self.name,
                          method='MCMC',
                          nsubjects=data['N'],
                          nparams=len(self.generative_model.paramnames['code']))

        results.paramnames = self.generative_model.paramnames['long']

        model_code = self.generative_model.model

        # Compile generative model with Stan
        sm = pystan.StanModel(model_code=model_code,
                              verbose=compile_verbose)

        # Sample from generative model
        stanfit = sm.sampling(data=data,
                              chains=chains,
                              iter=n_iterations,
                              warmup=warmup,
                              thin=thin,
                              seed=seed,
                              init=init,
                              sample_file=sample_file,
                              verbose=sampling_verbose,
                              algorithm=algorithm,
                              control=control,
                              n_jobs=n_jobs)

        # Create summary dataframe
        summary_data = stanfit.summary()['summary']
        summary_colnames = stanfit.summary()['summary_colnames']
        summary_rownames = stanfit.summary()['summary_rownames']
        stan_summary = pd.DataFrame(data=summary_data,
                                    columns=summary_colnames,
                                    index=summary_rownames)

        # Extract parameter estimates for ease of manipulation with fitrfit
        param_codes = self.generative_model.paramnames['code']
        param_est = stanfit.extract(pars=param_codes)

        # Get expected parameter estimates (subject-level) into params array
        param_idx = 0
        for k in self.generative_model.paramnames['code']:
            results.params[:, param_idx] = np.mean(param_est[k], axis=0)
            param_idx += 1

        # Populate results.stanfit dict with Stan related objects
        results.stanfit = {
            'stanfit' : stanfit,
            'summary' : stan_summary
        }

        return results

#
# TODO: VB AND GAUSSIAN PROCESSES
#
#
#class VB(object):
#    def __init__(self):
#        pass
#
#class GP(object):
#    def __init__(self):
#        pass

#===============================================================================
#
#   FITRFITS
#       Objects storing the optimization data
#
#===============================================================================

class fitrfit(object):
    """
    Class representing the results of a fitrmodel fitting.

    Attributes
    ----------
    name : str
        Model identifier. We suggest using free-parameters as identifiers
    method : str
        Method employed in optimization.
    nsubjects : int
        Number of subjects fitted.
    nparams : int
        Number of free parameters in the fitted model.
    params : ndarray(shape=(nsubjects, nparams))
        Array of parameter estimates
    errs : ndarray(shape=(nsubjects, nparams))
        Array of parameter estimate errors
    nlogpost : ndarray(shape=(nsubjects))
        Subject level negative log-posterior probability
    nloglik : float
        Subject level negative log-likelihood
    LME : float
        Log-model evidence
    BIC : ndarray(shape=(nsubjects))
        Subject-level Bayesian Information Criterion
    AIC : ndarray(shape=(nsubjects))
        Subject-level Aikake Information Criterion
    summary : DataFrame
        Summary of means and standard deviations for each free parameter, as well as negative log-likelihood, log-model-evidence, BIC, and AIC for the model

    Methods
    -------
    set_paramnames(params) :
        Sets names of RL parameters to the fitrfit object
    plot_ae(actual, show_figure=True, save_figure=False, filename='actual-estimate.pdf') :
        Plots estimated parameters against actual simulated parameters
    plot_fit_ts(show_figure=True, save_figure=False, filename='fit-stats.pdf') :
        Plots the evolution of log-likelihood, log-model-evidence, AIC, and BIC over optimization iterations
    param_hist(show_figure=True, save_figure=False, filename='param-hist.pdf') :
        Plots hitograms of parameters in the model
    summary_table(write_csv=False, filename='summary-table.csv', delimiter=',') :
        Writes a CSV file with summary statistics from the present model
    """
    def __init__(self, method, nsubjects, nparams, name=None):
        self.name = name
        self.method = method
        self.nsubjects = nsubjects
        self.nparams = nparams
        self.params = np.zeros([nsubjects, nparams])
        self.paramnames = []

        if method == 'MCMC':
            self.stanfit = None
        else:
            self.hess = np.zeros([nparams, nparams, nsubjects])
            self.hess_inv = np.zeros([nparams, nparams, nsubjects])
            self.errs = np.zeros([nsubjects, nparams])
            self.nlogpost = np.zeros(nsubjects) + 1e7
            self.nloglik = np.zeros(nsubjects)
            self.LME = np.zeros(nsubjects)
            self.BIC = np.zeros(nsubjects)
            self.AIC = np.zeros(nsubjects)
            self.ts_LME = []
            self.ts_nLL = []
            self.ts_BIC = []
            self.ts_AIC = []

    def set_paramnames(self, params):
        """
        Sets the names of the RL parameters to the fitrfit object

        Parameters
        ----------
        params : list
            List of parameters from the rlparams module
        """
        for i in range(len(params)):
            self.paramnames.append(params[i].name)

    def plot_ae(self, actual, show_figure=True, save_figure=False, filename='actual-estimate.pdf'):
        """
        Plots actual parameters (if provided) against estimates

        Parameters
        ----------
        actual : ndarray(shape=(nsubjects, nparams))
            Array of actual parameters from a simulation
        show_figure : bool
            Whether to show figure output
        save_figure : bool
            Whether to save the figure to disk
        filename : str
            The file name to be output
        """
        nparams = np.shape(self.params)[1]
        fig, ax = plt.subplots(1, nparams)
        for i in range(nparams):
            maxval = np.maximum(np.max(actual[:,i]), np.max(self.params[:,i]))
            minval = np.minimum(np.min(actual[:,i]), np.min(self.params[:,i]))
            ax[i].scatter(actual[:,i], self.params[:,i])
            ax[i].plot(np.linspace(minval, maxval, 100), np.linspace(minval, maxval, 100), c='k', ls='--')
            ax[i].set_xlabel('Actual')
            ax[i].set_ylabel('Estimate')
            ax[i].set_title(self.paramnames[i])
            ax[i].set_ylim([minval, maxval])
            ax[i].set_xlim([minval, maxval])

        if save_figure is True:
            plt.savefig(filename)

        if show_figure is True:
            plt.show()

        return

    def plot_fit_ts(self, show_figure=True, save_figure=False, filename='fit-stats.pdf'):
        """
        Plots the log-model-evidence, BIC, and AIC over optimization iterations

        Parameters
        ----------
        show_figure : bool
            Whether to show figure output
        save_figure : bool
            Whether to save the figure to disk
        filename : str
            The file name to be output
        """
        n_opt_steps = len(self.ts_LME)
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))

        #Log-Likelihood
        ax[0].plot(np.arange(n_opt_steps), self.ts_nLL, lw=1.5, c='k')
        ax[0].scatter(np.arange(n_opt_steps), self.ts_nLL, c='k')
        ax[0].set_xlabel('Optimization step')
        ax[0].set_title('Negative Log-Likelihood\n')
        ax[0].set_xlim([0, n_opt_steps])

        #LME
        ax[1].plot(np.arange(n_opt_steps), self.ts_LME, lw=1.5, c='k')
        ax[1].scatter(np.arange(n_opt_steps), self.ts_LME, c='k')
        ax[1].set_xlabel('Optimization step')
        ax[1].set_title('Log Model Evidence (LME)\n')
        ax[1].set_xlim([0, n_opt_steps])

        #BIC
        ax[2].plot(np.arange(n_opt_steps), self.ts_BIC, lw=1.5, c='k')
        ax[2].scatter(np.arange(n_opt_steps), self.ts_BIC, c='k')
        ax[2].set_xlabel('Optimization step')
        ax[2].set_title('Bayesian Information Criterion\n')
        ax[2].set_xlim([0, n_opt_steps])

        #AIC
        ax[3].plot(np.arange(n_opt_steps), self.ts_AIC, lw=1.5, c='k')
        ax[3].scatter(np.arange(n_opt_steps), self.ts_AIC, c='k')
        ax[3].set_xlabel('Optimization step')
        ax[3].set_title('Aikake Information Criterion\n')
        ax[3].set_xlim([0, n_opt_steps])

        if save_figure is True:
            plt.savefig(filename, bbox_inches='tight')

        if show_figure is True:
            plt.show()

        return

    def param_hist(self, show_figure=True, save_figure=False, filename='param-hist.pdf'):
        """
        Plots histograms of the parameter estimates

        Parameters
        ----------
        show_figure : bool
            Whether to show figure output
        save_figure : bool
            Whether to save the figure to disk
        filename : str
            The file name to be output

        """
        nparams = np.shape(self.params)[1]

        fig, ax = plt.subplots(1, nparams, figsize=(nparams*5, 7))
        for i in range(0, nparams):
            n, bins, patches = ax[i].hist(self.params[:,i], normed=1)
            y = mvn.pdf(bins, np.mean(self.params[:,i]), np.std(self.params[:,1]))
            ax[i].plot(bins, y, 'r--', lw=1.5)
            ax[i].set_title(self.paramnames[i] + '\n')

        if save_figure is True:
            plt.savefig(filename, bbox_inches='tight')

        if show_figure is True:
            plt.show()

        return

    def trace_plot(self, figsize=[8, 8], show_figure=True, save_figure=False, filename='fitr-mcstan-traceplot.pdf'):
        """
        Easy wrapper for Stan Traceplot

        Parameters
        ----------
        figsize : array [width in inches, height in inches]
            Controls figure size
        show_figure : bool
            Whether to show figure output
        save_figure : bool
            Whether to save the figure to disk
        filename : str
            The file name to be output

        """

        if self.method != 'MCMC':
            print('ERROR: Traceplot can only be created for MCMC results.')
        else:
            # Ignore the annoying warning about tight layout
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                mcplot = self.stanfit['stanfit'].traceplot()
                mcplot.set_size_inches(figsize[0], figsize[1])
                mcplot.set_tight_layout(tight=True)

                if save_figure is True:
                    plt.savefig(filename, bbox_inches='tight')

                if show_figure is True:
                    plt.show()

        return

    def ae_metrics(self, actual, matches=None):
        """
        Computes metrics summarizing the ability of the model to fit data generated from a known model

        Parameters
        ----------
        matches : list
            List consisting of [rlparams object, column index in `actual`, column index in estimates]. Ensures comparisons are being made between the same parameters, particularly when the models have different numbers of free parameters.

        Returns
        -------
        DataFrame
            Including summary statistics of the parameter matching
        """

        # [TODO] Complete this function

        pass

    def summary_table(self):
        """
        Generates a table summarizing the model-fitting results
        """

        summary_dict = {}

        pmeans = np.mean(self.params, axis=0)
        psd = np.std(self.params, axis=0)

        for i in range(self.nparams):
            summary_dict[self.paramnames[i]] = [pmeans[i], psd[i]]

        summary_dict['Neg-LL'] = [np.mean(self.nloglik),
                                  np.std(self.nloglik)]

        summary_dict['LME'] = [np.mean(self.LME),
                               np.std(self.LME)]

        summary_dict['BIC'] = [np.mean(self.BIC),
                               np.std(self.BIC)]

        summary_dict['AIC'] = [np.mean(self.AIC),
                               np.std(self.AIC)]

        index_labels = ['Mean', 'SD']
        column_labels = self.paramnames + ['Neg-LL', 'LME', 'BIC', 'AIC']

        self.summary = pd.DataFrame(summary_dict,
                                    index=index_labels,
                                    columns=column_labels)
