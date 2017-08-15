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
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

from ..criticism.model_evaluation import AIC, LME

class ModelFitResult(object):
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
    paramnames : list
        List of parameter names

    Methods
    -------
    set_paramnames(params)
        Sets names of RL parameters to the fitrfit object
    plot_ae(actual, save_figure=False, filename='actual-estimate.pdf')
        Plots estimated parameters against actual simulated parameters
    summary_table(write_csv=False, filename='summary-table.csv', delimiter=',')
        Writes a CSV file with summary statistics from the present model
    """
    def __init__(self, method, nsubjects, nparams, name=None):
        self.name = name
        self.method = method
        self.nsubjects = nsubjects
        self.nparams = nparams
        self.params = np.zeros([nsubjects, nparams])
        self.paramnames = []

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

    def plot_ae(self, actual, save_figure=False, filename='actual-estimate.pdf'):
        """
        Plots actual parameters (if provided) against estimates

        Parameters
        ----------
        actual : ndarray(shape=(nsubjects, nparams))
            Array of actual parameters from a simulation
        save_figure : bool
            Whether to save the figure to disk
        filename : str
            The file name to be output
        """
        nparams = np.shape(self.params)[1]
        fig, ax = plt.subplots(1, nparams, figsize=(nparams*5, 5))
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
            plt.savefig(filename, bbox_inches='tight')

        return fig

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

class OptimizationFitResult(ModelFitResult):
    """
    Results of model fitting with optimization methods

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
    paramnames : list
        List of parameter names
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
    plot_fit_ts(save_figure=False, filename='fit-stats.pdf') :
        Plots the evolution of log-likelihood, log-model-evidence, AIC, and BIC over optimization iterations
    param_hist(save_figure=False, filename='param-hist.pdf') :
        Plots hitograms of parameters in the model
    summary_table(write_csv=False, filename='summary-table.csv', delimiter=',')
        Writes a CSV file with summary statistics from the present model

    """
    def __init__(self, method, nsubjects, nparams, name):
        ModelFitResult.__init__(self,
                                method=method,
                                nsubjects=nsubjects,
                                nparams=nparams,
                                name=name)

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

    def plot_fit_ts(self, save_figure=False, filename='fit-stats.pdf'):
        """
        Plots the log-model-evidence, BIC, and AIC over optimization iterations

        Parameters
        ----------
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

        return fig

    def param_hist(self, save_figure=False, filename='param-hist.pdf'):
        """
        Plots histograms of the parameter estimates

        Parameters
        ----------
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

        return fig

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

class MCMCFitResult(ModelFitResult):
    """
    Results of model fitting with MCMC

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
    paramnames : list
        List of parameter names
    stanfit :
        Stan fit object
    summary : pandas.DataFrame
        Summary of the MCMC fit results

    Methods
    -------
    get_paramestimates(self, FUN=np.mean)
        Extracts parameter estimates
    trace_plot(self, figsize=None, save_figure=False, filename='fitr-mcstan-traceplot.pdf')
        Trace plot for fit results
    """
    def __init__(self, method, nsubjects, nparams, name):
        ModelFitResult.__init__(self,
                                method=method,
                                nsubjects=nsubjects,
                                nparams=nparams,
                                name=name)

        self.param_codes = []

        self.stanfit = None
        self.summary = None

    def make_summary(self):
        """
        Creates summary of Stan fitting results
        """

        # Create summary dataframe
        summary_data = self.stanfit.summary()['summary']
        summary_colnames = self.stanfit.summary()['summary_colnames']
        summary_rownames = self.stanfit.summary()['summary_rownames']
        self.summary = pd.DataFrame(data=summary_data,
                                    columns=summary_colnames,
                                    index=summary_rownames)

    def get_paramestimates(self, FUN=np.median):
        """
        Extracts parameter estimates

        Parameters
        ----------
        FUN : {numpy.mean, numpy.median}

        """
        param_est = self.stanfit.extract(pars=self.param_codes)

        # Get expected parameter estimates (subject-level) into params array
        param_idx = 0
        for k in self.param_codes:
            self.params[:, param_idx] = FUN(param_est[k], axis=0)
            param_idx += 1

    def trace_plot(self, figsize=None, save_figure=False, filename='fitr-mcstan-traceplot.pdf'):
        """
        Easy wrapper for Stan Traceplot

        Parameters
        ----------
        figsize : (optional) list [width in inches, height in inches]
            Controls figure size
        save_figure : bool
            Whether to save the figure to disk
        filename : str
            The file name to be output
        """
        if figsize is None:
            figsize = [8, 8]

        # Ignore the annoying warning about tight layout
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            mcplot = self.stanfit.traceplot()
            mcplot.set_size_inches(figsize[0], figsize[1])
            mcplot.set_tight_layout(tight=True)

            if save_figure is True:
                plt.savefig(filename, bbox_inches='tight')

        return mcplot
