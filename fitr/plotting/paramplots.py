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
Various plotting functions for parameter estimates

Module Documentation
--------------------
"""

import numpy as np
import matplotlib.pyplot as plt

def param_scatter(X, Y, paramnames=None, xlabel='Parameter Value', ylabel='y value', ylim=None, alpha=0.5, figsize=None, save_figure=False, filename='param-scatter.pdf'):
    """
    Plots a value against parameter estimates for each parameter

    Parameters
    ----------
    X : ndarray(shape=(nsubjects, nparams))
        Parameter array
    Y : ndarray(shape=nsubjects)
        Value to be plotted against parameters
    paramnames : (optional) list
        Parameter names (will be the title for each plot)
    xlabel : str
        Label for x axis
    ylabel : str
        Label for y axis
    ylim : (optional) tuple (min, max)
        Y-axis limits
    alpha : 0 < float < 1
        Transparency of scatter points
    figsize : (optional) tuple (width, height)
    save_figure : bool
        Whether to save the plot
    filename : str
        Path to which to plot the figure

    Returns
    -------
    matplotlib.pyplot.figure

    """
    nparams = np.shape(X)[1]

    if figsize is None:
        figsize = (4*nparams, 4)

    if paramnames is None:
        paramnames = np.arange(nparams)

    fig, ax = plt.subplots(nrows=1, ncols=nparams, figsize=figsize)
    for i in range(nparams):
        ax[i].scatter(X[:, i], Y, alpha=alpha)
        ax[i].set_title(str(paramnames[i]))
        ax[i].set_xlabel(xlabel)
        if ylim is not None:
            ax[i].set_ylim(ylim)
        if i == 0:
            ax[i].set_ylabel(ylabel)

    plt.tight_layout()

    if save_figure is True:
        plt.savefig(filename, bbox_inches='tight')

    return fig
