import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, loglik_func, params):
        self.loglik_func = loglik_func
        self.params = params

    def fit(self, data, method='EM'):

        if method=='EM':
            opt = EM(loglik_func=self.loglik_func, params=self.params)
            results = opt.fit(data=data)
        elif method=='MLE':
            opt = MLE(loglik_func=self.loglik_func, params=self.params)
            results = opt.fit(data=data)

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

    References
    ----------
    Huys et al. (2011)
    """
    def __init__(self, loglik_func, params):
        from scipy.stats import multivariate_normal as mvn

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
        """
        from scipy.optimize import minimize

        # Instantiate the results object
        nsubjects = len(data)
        results = fitrfit(method='Expectation-Maximization',
                          nsubjects=nsubjects,
                          nparams=self.nparams)

        convergence = False
        opt_iter = 1
        sum_nlogpost = 0 # Monitor total neg-log-posterior for convergence
        while convergence == False and opt_iter < n_iterations:
            for i in range(nsubjects):
                print('ITERATION: '          + str(opt_iter) +
                      ' | FITTING SUBJECT: ' + str(i+1))

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
                    results.hess_inv[:, :, i] = res.hess_inv
                    results.nloglik[i]  = self.loglik_func(params=trans_UC(res.x, rng=self.param_rng), states=S, actions=A, rewards=R)

            # If the group level posterior probability has converged, stop
            if np.abs(np.sum(results.nlogpost) - sum_nlogpost) < c_limit:
                convergence = True
            else:
                # Update the running model log-posterior probability
                sum_nlogpost = np.sum(results.nlogpost)
                # Optimize group level hyperparameters
                self.group_level_estimate(param_est=results.params,
                                          hess_inv=results.hess_inv)
                opt_iter += 1

        # Transform data to constrained space
        for i in range(nsubjects):
            results.params[i,:] = trans_UC(results.params[i,:], self.param_rng)

        return results

    def logposterior(self, x, states, actions, rewards):
        """ Represents the log-posterior probability function """

        lp = self.loglik_func(params=trans_UC(x, rng=self.param_rng), states=states, actions=actions, rewards=rewards) + self.prior.logpdf(x, mean=self.mu, cov=self.cov)

        return lp

    def group_level_estimate(self, param_est, hess_inv):
        """
        Updates the group-level hyperparameters
        """
        print('==== UPDATING HYPERPARAMETERS ====')

        nsubjects = np.shape(param_est)[0]

        self.mu = np.mean(param_est, axis=0)
        self.cov = np.zeros([self.nparams, self.nparams])

        for i in range(nsubjects):
            self.cov = self.cov + (np.outer(param_est[i,:], param_est[i,:]) + hess_inv[:, :, i]) - np.outer(self.mu, self.mu)

        self.cov = self.cov/nsubjects


class MLE(object):
    def __init__(self, loglik_func, params):
        self.loglik_func = loglik_func
        self.params = params
        self.nparams = len(params)

    def fit(self, data):
        pass

class EmpiricalPriors(object):
    def __init__(self):
        pass

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
    def __init__(self, method, nsubjects, nparams):

        self.method = method
        self.nsubjects = nsubjects
        self.nparams = nparams
        self.params = np.zeros([nsubjects, nparams])
        self.paramnames = None
        self.hess_inv = np.zeros([nparams, nparams, nsubjects])
        self.errs = np.zeros([nsubjects, nparams])
        self.nlogpost = np.zeros(nsubjects) + 1e7
        self.nloglik = np.zeros(nsubjects)
        self.LME = None
        self.BIC = np.zeros(nsubjects)
        self.AIC = np.zeros(nsubjects)

    def plot_ae(self, actual):
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

        plt.show()


#===============================================================================
#
#   PARAMETERS
#       Objects denoting parameters for simple use
#
#===============================================================================

class Param(object):
    def __init__(self, name=None, rng=None):
        self.name = name
        self.range = rng

class LearningRate(Param):
    def __init__(self, name='Learning Rate', rng='unit'):
        self.name = name
        self.rng  = rng

class ChoiceRandomness(Param):
    def __init__(self, name='Choice Randomness', rng='pos'):
        self.name = name
        self.rng  = rng

#===============================================================================
#
#   UTILITY FUNCTIONS
#       Functions used across fitr modules
#
#===============================================================================

def softmax(x):
    xmax = np.max(x)
    return np.exp(x-xmax)/np.sum(np.exp(x-xmax))

def logsumexp(x):
    """
    Numerically stable logsumexp.
    """
    xmax = np.max(x)
    y = xmax + np.log(np.sum(np.exp(x-xmax)))
    return y

def trans_UC(values_U, rng):
    'Transform parameters from unconstrained to constrained space.'
    if rng[0] == 'all_unc':
        return values_U
    values_T = []
    for value, rng in zip(values_U, rng):
        if rng   == 'unit':  # Range: 0 - 1.
            if value < -16.:
                value = -16.
            values_T.append(1./(1. + np.exp(-value)))  # Don't allow values smaller than 1e-
        elif rng   == 'half':  # Range: 0 - 0.5
            if value < -16.:
                value = -16.
            values_T.append(0.5/(1. + np.exp(-value)))  # Don't allow values smaller than 1e-7
        elif rng == 'pos':  # Range: 0 - inf
            if value > 16.:
                value = 16.
            values_T.append(np.exp(value))  # Don't allow values bigger than ~ 1e7.
        elif rng == 'unc': # Range: - inf - inf.
            values_T.append(value)
    return np.array(values_T)

def BIC(x):
    """ Bayesian information criterion """
    pass

def AIC(x):
    """ Aikake information criterion """
    pass
