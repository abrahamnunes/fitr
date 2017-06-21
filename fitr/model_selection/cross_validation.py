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
"""
Functions for cross validation
"""

import numpy as np
import matplotlib.pyplot as plt

from ..plotting import param_scatter

class LOACV(object):
    """
    Look-one-ahead cross validation

    Attributes
    ----------
    cv_func : loacv function
        A look-one-ahead cross-validation function from a Fitr model
    results : LookOneAheadCVResult
        Stores results of the cross validation
    """
    def __init__(self, cv_func):
        self.cv_func = cv_func
        self.results = None

    def run(self, params, data):
        """
        Runs the Look-One-Ahead cross validation

        Parameters
        ----------
        params : ndarray(shape=(nsubjects, nparams))
            Array of parameters
        data : dict
            Behavioural data in Fitr OptimizationData format
        """
        nsubjects, nparams = np.shape(params)

        self.results = LookOneAheadCVResult(params=params)

        for i in range(nsubjects):
            loocv_data = self.cv_func(params=params[i, :],
                                      states=data[i]['S'],
                                      actions=data[i]['A'],
                                      rewards=data[i]['R'])

            # Add to LookOneAheadCVResult
            self.results.accuracy.append(loocv_data['acc'])
            self.results.nLL.append(-loocv_data['LL'])
            self.results.raw[i] = loocv_data



class LookOneAheadCVResult(object):
    """
    Stores and manipulates results of a Look-One-Ahead cross-validation run

    Attributes
    ----------
    nsubjects : dict
        Dictionary of
    accuracy : dict
        Dictionary of accuracy values (overall and by subject)
    raw : dict
        Dictionary
    """
    def __init__(self, params):
        self.params = params
        self.nsubjects = np.shape(params)[0]
        self.nparams = np.shape(params)[1]
        self.accuracy = []
        self.nLL = []

        self.raw = {}

    def accuracy_maplot(self, save_figure=False, filename='accuracy-maplot.pdf', figsize=None):
        """
        Plots moving average of accuracy

        Parameters
        ----------
        save_figure : bool
            Whether to save the plot
        filename : str
            Name of the file to which figure will be saved
        figsize : (optional) tuple (width, height)
            The size of the figure

        """
        if figsize is None:
            figsize = (8, 8)

        fig, ax = plt.subplots(figsize=figsize)

        for i in range(self.nsubjects):
            subj_data = self.raw[i]
            ntrials = np.size(subj_data['A_match'])

            mov_avg = np.zeros(ntrials)
            for t in range(1, ntrials):
                mov_avg[t] = np.mean(subj_data['A_match'][:t])

            ax.plot(np.arange(ntrials-1), mov_avg[1:])

        ax.set_title('Look-One-Ahead CV Accuracy')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Accuracy')

        if save_figure is True:
            plt.savefig(filename, bbox_inches='tight')

        return fig

    def accuracy_hist(self, save_figure=False, filename='accuracy-hist.pdf', figsize=None):
        """
        Plots moving average of accuracy

        Parameters
        ----------
        save_figure : bool
            Whether to save the plot
        filename : str
            Name of the file to which figure will be saved
        figsize : (optional) tuple (width, height)
            The size of the figure

        """
        if figsize is None:
            figsize = (8, 8)

        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(self.accuracy, normed=1, edgecolor='k')

        ax.set_xlim([0, 1])
        ax.set_title('Look-One-Ahead CV Accuracy')
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Proportion')

        if save_figure is True:
            plt.savefig(filename, bbox_inches='tight')

        return fig

    def accuracy_param_scatter(self, paramnames=None, ylim=None, alpha=0.5, save_figure=False, filename='accuracy-param-scatter.pdf', figsize=None):
        """
        Plots accuracy against parameter values. Helpful to visually inspect the effects of various parameters on cross-validation accuracy

        Parameters
        ----------
        paramnames : (optional) list
            List of parameter names in strings
        ylim : (optional) tuple (min, max)
            Y-axis limits
        alpha : 0 < float < 1
            Transparency of the plot points
        save_figure : bool
            Whether to save the plot
        filename : str
            Name of the file to which figure will be saved
        figsize : (optional) tuple (width, height)
            The size of the figure

        Returns
        -------
        matplotlib.pyplot.figure

        """
        fig = param_scatter(X=self.params,
                            Y=self.accuracy,
                            ylabel='Accuracy',
                            ylim=ylim,
                            paramnames=paramnames,
                            alpha=alpha,
                            figsize=figsize,
                            save_figure=save_figure,
                            filename=filename)

        return fig
