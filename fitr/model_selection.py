"""
Functions for model selection/comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')

from scipy.special import digamma as psi
from scipy.special import gammaln


class ModelSelectionResult(object):
    """
    Object containing results of model selection

    Attributes
    ----------
    modelnames : list
        List of strings labeling models
    xp : ndarray
        Exceedance probabilities for each model
    pxp : ndarray
        Protected exceedance probabilities for each model
    BIC : ndarray
        Bayesian information criterion measures for each model
    AIC : ndarray
        Aikake information criterion measures for each model

    """
    def __init__(self, method):
        self.modelnames = []

        if method=='BMS':
            self.xp = []
            self.pxp = []

        if method=='BIC':
            self.BIC = []

        if method=='AIC':
            self.AIC = []

        if method=='All':
            self.xp = []
            self.pxp = []
            self.BIC = []
            self.AIC = []

    def plot(self, statistic, show_figure=True, save_figure=False, filename='modelselection-plot.pdf', figsize=(10, 10)):
        if statistic=='pxp':
            bar_height = self.pxp
            plot_title = 'Protected Exceedance Probabilities'
            plot_ylabel = 'Probability'
        elif statistic=='xp':
            bar_height = self.xp
            plot_title = 'Exceedance Probabilities'
            plot_ylabel = 'Probability'
        elif statistic=='BIC':
            bar_height = self.BIC
            plot_title = 'Bayesian Information Criterion'
            plot_ylabel = 'BIC'
        elif statistic=='AIC':
            bar_height = self.AIC
            plot_title = 'Aikake Information Criterion'
            plot_ylabel = 'AIC'

        width = 0.7
        ind = np.arange(len(self.modelnames))

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.bar(ind, bar_height, width=width, align="center")
        ax.set_ylabel(plot_ylabel)
        ax.set_xticks(ind)
        ax.set_xticklabels(self.modelnames)
        ax.set_title(plot_title)

        if save_figure is True:
            plt.savefig(filename)

        if show_figure is True:
            plt.show()

class BIC(object):
    """
    Model comparison with Bayesian Information Criterion

    Attributes
    ----------
    modelfits : list
        List of fitrfit objects from completed model fitting
    """
    def __init__(self, model_fits):
        self.modelfits = model_fits

    def run(self):
        """
        Runs model comparison by Bayesian Information Criterion
        """
        results = ModelSelectionResult(method='BIC')

        for i in range(len(self.modelfits)):
            results.BIC.append(self.modelfits[i].name)

        return results

class AIC(object):
    """
    Model comparison with Aikake Information Criterion

    Attributes
    ----------
    modelfits : list
        List of fitrfit objects from completed model fitting
    """
    def __init__(self, model_fits):
        self.modelfits = model_fits

    def run(self):
        """
        Runs model comparison by Aikake Information Criterion
        """
        results = ModelSelectionResult(method='AIC')

        for i in range(len(self.modelfits)):
            results.AIC.append(self.modelfits[i].name)

        return results

class BMS(object):
    """
    Bayesian model selection

    Attributes
    ----------
    modelfits : list
        List of fitrfit objects from completed model fitting
    nmodels : int
        Number of models to be compared
    nsubjects: int
        Number of subjects in the sample. (Must be equal across all fits).
    c_limit : float
        Threshold at which to stop model comparison

    Methods
    -------
    run
        Runs the Bayesian model selection algorithm
    dirichlet_exceedance
        Computes exceedance probabilities for a Dirichlet distribution


    References
    ----------
    [1] Rigoux, L. et al. (2014) Neuroimage 84, 971–985
    [2] Samuel Gershman's _mfit_ package (on GitHub)
    """
    def __init__(self, model_fits, c_limit=10e-100):
        self.modelfits = model_fits

        # Extract number of models and number of subjects
        self.nmodels = len(self.modelfits)
        self.nsubjects = self.modelfits[0].nsubjects
        self.c_limit = c_limit

    def run(self):
        """
        Runs Bayesian model selection algorithm

        Returns
        -------
        ModelComparisonResult :
            Object representing model comparison results
        """

        bms_results = {}

        # Preallocate memory
        u = np.zeros([self.nsubjects, self.nmodels])
        g = np.zeros([self.nsubjects, self.nmodels])
        a = np.zeros([self.nsubjects, self.nmodels])
        lme = np.zeros([self.nsubjects, self.nmodels])

        # Initialize Dirichlet priors
        alpha0 = np.ones(self.nmodels)
        alpha = alpha0

        bms_iter = 0
        prediction_error = 1

        print('===== STARTING BAYESIAN MODEL SELECTION =====\n' +
              'Number of models: ' + str(self.nmodels) + '\n' +
              'Subjects: ' + str(self.nsubjects) + '\n' +
              'Convergence limit: ' + str(self.c_limit) + '\n' +
              '=============================================\n')

        while prediction_error > self.c_limit:
            bms_iter += 1
            print('[BMS] ITERATION: ' + str(bms_iter))

            for i in range(self.nsubjects):
                log_u = np.zeros(self.nmodels)
                for j in range(self.nmodels):
                    test_lme = self.modelfits[j].LME[i]

                    # If Hessian was degenerate, use BIC approximation of LME
                    lmefinite = np.isfinite(test_lme)
                    lmecomplex = np.iscomplex(test_lme)
                    if lmefinite is False or lmecomplex is True:
                        lme[i, j] = -0.5*self.modelfits[j].BIC[i]
                        - self.modelfits[j].nparams*np.log(2*np.pi)
                    else:
                        lme[i, j] = self.modelfits[j].LME[i]

                    log_u[j] = lme[i, j] + psi(alpha[j]) - psi(np.sum(alpha))

                u[i, :] = np.exp(log_u - np.max(log_u))
                g[i, :] = u[i, :]/np.sum(u[i, :])
                a[i, :] = np.random.multinomial(1, pvals=g[i, :])

            beta = np.sum(a, axis=0)
            alpha_previous = alpha
            alpha = alpha0 + beta

            # Test for convergence
            prediction_error = np.linalg.norm(alpha - alpha_previous)

        # Find expected values of model probabilities
        bms_results = {
                'pms': g,
                'r': alpha/np.sum(alpha),
                'xp': self.dirichlet_exceedance(alpha),
                'pxp': []
                }

        posterior = {
            'a': alpha,
            'r': g.T
        }
        priors = {'a': alpha0}
        bor = self.BOR(L=lme.T, posterior=posterior, priors=priors)
        bms_results['pxp'] = np.dot(bms_results['xp'], (1-bor))
        + bor/self.nmodels

        # Store data in ModelSelectionResult object
        results = ModelSelectionResult(method='BMS')
        for i in range(len(self.modelfits)):
            results.modelnames.append(self.modelfits[i].name)
        results.pxp = bms_results['pxp']
        results.xp = bms_results['xp']

        return results

    def dirichlet_exceedance(self, alpha):
        """
        Computes exceedance probabilities for a Dirichlet distribution

        Parameters
        ----------
        alpha :

        Returns
        -------
        xp :
            Exceedance probabilities

        References
        ----------
        [1] Samuel Gershman's _mfit_ package (on GitHub)
        """

        nsamples = 1e6
        K = len(alpha)

        # Perform sampling in blocks
        blk = np.ceil(nsamples*K*8/(2**28))
        blk = np.floor(nsamples/blk * np.ones(blk))
        blk[-1] = nsamples - np.sum(blk[:-1])

        xp = np.zeros(K)
        for i in range(0, len(blk)):
            # Sample from univariate gamma then normalize
            r = np.zeros([int(blk[i]), K])
            for k in range(0, K):
                r[:, k] = np.random.gamma(alpha[k], 1, size=int(blk[i]))

            sr = np.sum(r, axis=1)
            for k in range(K):
                r[:, k] = r[:, k]/sr

            # Compute Exceedance probabilities:
            #   For any given model k1, compute probability that
            #    it is more likely than any other model k2 != k1
            j = np.argmax(r, axis=1)
            xp = xp + np.histogram(j, bins=K)[0]

        xp = xp/nsamples
        return xp

    def BOR(self, L, posterior, priors, C=None):
        """
        Computes Bayes Omnibus Risk (BOR)

        Parameters
        ----------
        L :
        posterior:
        priors:
        C:

        Returns
        -------
        bor :
            Bayesian omnibus risk

        References
        ----------
        [1] Samuel Gershman's _mfit_ package (on GitHub)
        """

        if C is None:
            options = {'families': False}

            # Evidence of null (equal model frequencies)
            F0 = self.FE_null(L, options)[0]
        else:
            options = {
                'families': True,
                'C': C
            }

            # Evidence of null (equal model frequencies) under family prior
            F0 = self.FE_null(L, options)[1]

        # Evidence of alternative
        F1 = self.FE(L=L, posterior=posterior, priors=priors)

        bor = 1/(1+np.exp(F1-F0))
        return bor

    def FE(self, L, posterior, priors):
        """
        Derives free energy for current approximate posterior distribution

        Parameters
        ----------
        L :
        posterior :
        priors :

        Returns
        -------
        F :
            Free energy of the current posterior

        References
        ----------
        [1] Rigoux L., Daunizeau J. _VBA Toolbox_
        """
        nmodels, nsubjects = np.shape(L)
        a0 = np.sum(posterior['a'])
        Elogr = psi(posterior['a']) - psi(np.sum(posterior['a']))
        Sqf = np.sum(gammaln(posterior['a']))
        - gammaln(a0) - np.sum((posterior['a']-1)*Elogr)
        Sqm = 0

        for i in range(nsubjects):
            Sqm = Sqm - np.sum(posterior['r'][:, i] *
                               np.log(posterior['r'][:, i]+np.spacing(1)))

        gln_sumprior = gammaln(np.sum(priors['a']))
        sum_glnprior = np.sum(gammaln(priors['a']))
        ELJ = gln_sumprior - sum_glnprior + np.sum((priors['a']-1)*Elogr)

        for i in range(nsubjects):
            for k in range(nmodels):
                ELJ = ELJ + posterior['r'][k, i]*(Elogr[k]+L[k, i])

        F = ELJ + Sqf + Sqm

        return F

    def FE_null(self, L, options):
        """
        Derives the free energy of the 'null' hypothesis

        Parameters
        ----------
        L :
        options :

        Returns
        -------
        F0m :
            Evidence for the null (i.e. equal probabilities) over models
        F0f :
            Evidence for the null (i.e. equal probabilities) over families

        References
        ----------
        [1] Rigoux L., Daunizeau J. _VBA Toolbox_
        """
        nmodels, nsubjects = np.shape(L)
        if options['families'] is True:
            f0 = np.dot(options['C'],
                        np.sum(options['C'],
                        axis=0))**(-1/(np.shape(options['C'])[1]))
            F0f = 0
        else:
            F0f = []

        F0m = 0
        for i in range(nsubjects):
            tmp = L[:, i] - np.max(L[:, i])
            g = np.exp(tmp)/np.sum(np.exp(tmp))
            for k in range(nmodels):
                lme_err = L[k, i]-np.log(nmodels)-np.log(g[k] + np.spacing(1))
                F0m = F0m + g[k]*lme_err
                if options['families'] is True:
                    logf0 = np.log(f0[k])
                    logGeps = np.log(g[k]+np.spacing(1))
                    F0f = F0f + g[k] * (L[k, i]-logGeps+logf0)

        return F0m, F0f
