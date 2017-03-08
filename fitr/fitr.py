import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as mvn

from .utils import trans_UC, BIC, AIC, LME

#===============================================================================
#
#   FITRMODEL
#       The highest level object representing a task model to be fit.
#       It is not necessary to use this object, since one could use the EM
#        object (lower level), but this method should be the simplest for
#        inexperienced users
#
#===============================================================================

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
    """
    def __init__(self, loglik_func, params, name=None):
        self.name = name
        self.loglik_func = loglik_func
        self.params = params

    def fit(self, data, method='EM', c_limit=0.01):
        """
        Runs model fitting

        Parameters
        ----------
        method : {'EM', 'MLE'}
            The inference algorithm to use.
        c_limit : float
            Limit at which convergence of log-posterior probability is determined

        Returns
        -------
        fitrfit : object
            Representation of the model fitting results
        """

        if method=='EM':
            opt = EM(loglik_func=self.loglik_func, params=self.params, name=self.name)
            results = opt.fit(data=data, c_limit=c_limit)
        elif method=='MLE':
            opt = MLE(loglik_func=self.loglik_func, params=self.params, name=self.name)
            results = opt.fit(data=data, c_limit=c_limit)

        return results

#===============================================================================
#
#   INFERENCE METHOD OBJECTS
#       - EM: Expectation-Maximization with Laplace Approximation
#       - MLE: Maximum Likelihood Estimate
#
#===============================================================================

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

    References
    ----------
    [1] Huys, Q. J. M., et al. (2011). Disentangling the roles of approach, activation and valence in instrumental and pavlovian responding. PLoS Computational Biology, 7(4).
    """
    def __init__(self, loglik_func, params, name=None):
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
        self.cov = np.eye(self.nparams)

    def fit(self, data, n_iterations=1000, c_limit=1, opt_algorithm='BFGS'):
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
        opt_algorithm : {'BFGS', 'L-BFGS-B', 'Nelder-Mead'}
            Algorithm to use for optimization

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

        convergence = False
        opt_iter = 1
        sum_nlogpost = 0 # Monitor total neg-log-posterior for convergence
        while convergence == False and opt_iter < n_iterations:
            for i in range(nsubjects):
                print('ITERATION: '          + str(opt_iter) +
                      ' | [E-STEP] SUBJECT: ' + str(i+1))

                # Extract subject-level data
                S = data[i]['S']
                A = data[i]['A']
                R = data[i]['R']

                # Construct the subject's negative log-posterior function
                _nlogpost = lambda x: -self.logposterior(x=x, states=S, actions=A, rewards=R)

                # Generate initial values
                x0 = np.random.normal(loc=0, scale=2, size=self.nparams)
                # Optimize
                res = minimize(_nlogpost, x0, method=opt_algorithm)

                # Update subject level data if logposterior improved
                if res.fun < results.nlogpost[i]:
                    results.nlogpost[i] = res.fun
                    results.params[i,:] = res.x
                    results.hess[:,:,i] = np.linalg.inv(res.hess_inv)
                    results.hess_inv[:,:,i] = res.hess_inv
                    results.nloglik[i]  = self.loglik_func(params=trans_UC(res.x, rng=self.param_rng), states=S, actions=A, rewards=R)
                    results.LME[i] = LME(-res.fun, self.nparams, np.linalg.inv(res.hess_inv))
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
                results.ts_BIC.append(np.sum(results.BIC))
                results.ts_AIC.append(np.sum(results.AIC))

                # Optimize group level hyperparameters
                print('\nITERATION: ' + str(opt_iter) + ' | [M-STEP]\n')
                self.group_level_estimate(param_est=results.params,
                                          hess_inv=results.hess_inv)
                opt_iter += 1

        # Transform data to constrained space
        for i in range(nsubjects):
            results.params[i,:] = trans_UC(results.params[i,:], self.param_rng)

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

    def group_level_estimate(self, param_est, hess_inv):
        """
        Updates the group-level hyperparameters

        Parameters
        ----------
        param_est : ndarray(shape=(nsubjects, nparams))
            Current parameter estimates for each subject
        hess_inv : ndarray(shape=(nparams, nparams, nsubjects))
            Inverse Hessian matrix estimate for each subject from the iteration with highest log-posterior probability
        """
        nsubjects = np.shape(param_est)[0]

        self.mu = np.mean(param_est, axis=0)
        self.cov = np.zeros([self.nparams, self.nparams])

        for i in range(nsubjects):
            self.cov = self.cov + (np.outer(param_est[i,:], param_est[i,:]) + hess_inv[:, :, i]) - np.outer(self.mu, self.mu)

        self.cov = self.cov/nsubjects
#
# TODO: ADD MLE, EMPIRICALPRIORS, MCMC, VB, AND GAUSSIAN PROCESSES
#
#class MLE(object):
#    def __init__(self, loglik_func, params):
#        self.loglik_func = loglik_func
#        self.params = params
#        self.nparams = len(params)
#
#    def fit(self, data):
#        pass
#
#class EmpiricalPriors(object):
#    def __init__(self):
#        pass
#
#class MCMC(object):
#    def __init__(self):
#        pass
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
#   FITRFIT
#       Object storing the optimization data
#
#===============================================================================

class fitrfit(object):
    """
    Class representing the results of a fitrmodel optimization.

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

    Methods
    -------
    transform :
        Transforms data to constrained or unconstrained space
    """
    def __init__(self, method, nsubjects, nparams, name=None):
        self.name = name
        self.method = method
        self.nsubjects = nsubjects
        self.nparams = nparams
        self.params = np.zeros([nsubjects, nparams])
        self.paramnames = []
        self.hess = np.zeros([nparams, nparams, nsubjects])
        self.hess_inv = np.zeros([nparams, nparams, nsubjects])
        self.errs = np.zeros([nsubjects, nparams])
        self.nlogpost = np.zeros(nsubjects) + 1e7
        self.nloglik = np.zeros(nsubjects)
        self.LME = np.zeros(nsubjects)
        self.BIC = np.zeros(nsubjects)
        self.AIC = np.zeros(nsubjects)
        self.ts_LME = []
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
            ax[i].plot(np.linspace(minval, maxval, 100), np.linspace(minval, maxval, 100))
            ax[i].scatter(actual[:,i], self.params[:,i])
            ax[i].set_xlabel('Actual')
            ax[i].set_ylabel('Estimate')
            ax[i].set_ylim([minval, maxval])
            ax[i].set_xlim([minval, maxval])

        if save_figure is True:
            plt.savefig(filename)

        if show_figure is True:
            plt.show()

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
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        #LME
        ax[0].plot(np.arange(n_opt_steps), self.ts_LME, lw=1.5, c='k')
        ax[0].scatter(np.arange(n_opt_steps), self.ts_LME, c='k')
        ax[0].set_xlabel('Optimization step')
        ax[0].set_title('Log Model Evidence (LME)\n')
        ax[0].set_xlim([0, n_opt_steps])

        #BIC
        ax[1].plot(np.arange(n_opt_steps), self.ts_BIC, lw=1.5, c='k')
        ax[1].scatter(np.arange(n_opt_steps), self.ts_BIC, c='k')
        ax[1].set_xlabel('Optimization step')
        ax[1].set_title('Bayesian Information Criterion\n')
        ax[1].set_xlim([0, n_opt_steps])

        #AIC
        ax[2].plot(np.arange(n_opt_steps), self.ts_AIC, lw=1.5, c='k')
        ax[2].scatter(np.arange(n_opt_steps), self.ts_AIC, c='k')
        ax[2].set_xlabel('Optimization step')
        ax[2].set_title('Aikake Information Criterion\n')
        ax[2].set_xlim([0, n_opt_steps])

        if save_figure is True:
            plt.savefig(filename)

        if show_figure is True:
            plt.show()

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

        fig, ax = plt.subplots(1, nparams)
        for i in range(0, nparams):
            n, bins, patches = ax[i].hist(self.params[:,i], normed=1)
            y = mvn.pdf(bins, np.mean(self.params[:,i]), np.std(self.params[:,1]))
            ax[i].plot(bins, y, 'r--', lw=1.5)
            ax[i].set_title(self.paramnames[i] + '\n')

        if save_figure is True:
            plt.savefig(filename)

        if show_figure is True:
            plt.show()
