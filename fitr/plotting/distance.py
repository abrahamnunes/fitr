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
Plotting functions for distance metrics

Module Documentation
--------------------
"""
import numpy as np
import matplotlib.pyplot as plt

def distance_scatter(X, Y, group_labels=None, xlab='', ylab='', show_figure=True, save_figure=False, figsize=None, figname='distance-scatter.pdf'):
    """
    Creates a scatterplot between two distance metrics, demonstrating group separation, if any.

    Parameters
    ----------
    group_labels : (optional)
    xlab : str
        X-axis label
    ylab : str
        Y-axis label
    show_figure : bool
        Whether to show the plot
    save_figure : bool
        Whether to save the figure
    figsize : (optional) list
        Controls figure size
    figname : str
        The name under which the plot should be saved
    """
    if figsize is None:
        figsize = [5, 5]

    fig, ax = plt.subplots(figsize=figsize)

    if group_labels is None:
        ax.scatter(X.flatten(), Y.flatten(), c='b')
    else:
        unique_labels = np.unique(group_labels)

        group_colour = np.zeros([len(group_labels), len(group_labels)])
        for i in range(len(group_labels)):
            for j in range(len(group_labels)):
                group_colour[i, j] = int(group_labels[i] == group_labels[j])


        x = X.flatten()
        y = Y.flatten()
        group_colour = group_colour.flatten()

        ax.scatter(x[group_colour == 0],
                   y[group_colour == 0],
                   c='r',
                   label='Out Group')
        ax.scatter(x[group_colour == 1],
                   y[group_colour == 1],
                   c='b',
                   label='In Group')

        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                   loc=3,
                   ncol=len(unique_labels),
                   mode="expand",
                   borderaxespad=0.)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)


    if save_figure is True:
        plt.savefig(figname, bbox_inches="tight")

    if show_figure is True:
        plt.show()


def distance_hist(X, group_labels, xlab='Distance', ylab='', normed=1, show_figure=True, save_figure=False, figsize=None, figname='distance-hist.pdf'):
    """
    Creates a histogram of within- and between-group distances.

    Parameters
    ----------
    group_labels : ndarray(size=n_labels)
        Vector of group labels for each participant represented
    xlab : str
        X-axis label
    ylab : str
        Y-axis label
    normed : 0 or 1 (default=1)
        Whether the histogram should be normalized
    show_figure : bool
        Whether to show the plot
    save_figure : bool
        Whether to save the figure
    figsize : (optional) list
        Controls figure size
    figname : str
        The name under which the plot should be saved
    """
    if figsize is None:
        figsize = [5, 5]

    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(group_labels)

    group_colour = np.zeros([len(group_labels), len(group_labels)])
    for i in range(len(group_labels)):
        for j in range(len(group_labels)):
            group_colour[i, j] = int(group_labels[i] == group_labels[j])


    x = X.flatten()
    group_colour = group_colour.flatten()

    ax.hist(x[group_colour == 0],
            normed=normed,
            facecolor='r',
            edgecolor='k',
            label='Out Group',
            alpha=0.4)
    ax.hist(x[group_colour == 1],
            normed=normed,
            facecolor='b',
            edgecolor='k',
            label='In Group',
            alpha=0.4)

    plt.legend()

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)


    if save_figure is True:
        plt.savefig(figname, bbox_inches="tight")

    if show_figure is True:
        plt.show()
