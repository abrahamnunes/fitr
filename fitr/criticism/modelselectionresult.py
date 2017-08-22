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
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')

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

    Methods
    -------
    plot(self, statistic, save_figure=False, filename='modelselection-plot.pdf', figsize=(10, 10))
        Plots the results of model selection (bars)

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

    def plot(self, statistic, save_figure=False, filename='modelselection-plot.pdf', figsize=(10, 10)):
        """
        Plots the results of model selection (bars)

        Parameters
        ----------
        statistic : {'pxp', 'xp', 'BIC', 'AIC'}
            Which statistic is desired for the bar plot
        save_figure : bool
            Whether to save the figure
        filename : str
            The desired filename for the plot (must end in appropriate extension)
        figsize : tuple, default (10, 10)

        """
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
            plt.savefig(filename, bbox_inches='tight')

        return fig
