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
Module containing embedding algorithms

Module Documentation
--------------------
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE as tsne

class Embedding(object):
    """
    Base embedding object

    Attributes
    ----------
    algorithm
        The embedding algorithm to be used
    embedding : ndarray(shape=(n_samples, n_components))

    Methods
    -------
    embed(self, data)
        Runs the embedding
    plot(self, group_labels=None, legend=True, figsize=None, show_figure=True, save_figure=False, figname='embedding.pdf')
        Plots the embedding
    """
    def __init__(self):
        self.algorithm = None
        self.embedding = None

    def embed(self, data):
        self.embedding = self.algorithm.fit_transform(data)

    def plot(self, group_labels=None, legend=True, figsize=None, show_figure=True, save_figure=False, figname='embedding.pdf'):
        """
        Plots the embedding. Currently only works for 2-component plots

        Parameters
        ----------
        group_labels : (optional) ndarray(shape=nsubjects)
            Labels for subject groups
        legend : bool
            Whether to plot a legend
        figsize : (optional) list [width in inches, height in inches]
            Controls figure size
        show_figure : bool
            Whether to show figure output
        save_figure : bool
            Whether to save the figure to disk
        filename : str
            The file name to be output
        """

        if figsize is None:
            figsize = [8, 8]

        n_samples = np.shape(self.embedding)[0]

        if group_labels is None:
            group_labels = np.zeros(n_samples)
            unique_labels = np.unique(group_labels)
        else:
            unique_labels = np.unique(group_labels)

        fig, ax = plt.subplots(figsize=figsize)
        c_list  = 'brgcmyk'
        for i in range(len(unique_labels)):
            ax.scatter(self.embedding[group_labels == unique_labels[i], 0],
                       self.embedding[group_labels == unique_labels[i], 1],
                       c=c_list[i],
                       label=unique_labels[i])

        ax.tick_params(axis='both',
                       left='off',
                       top='off',
                       right='off',
                       bottom='off',
                       labelleft='off',
                       labeltop='off',
                       labelright='off',
                       labelbottom='off')

        if len(unique_labels) > 1 and legend is True:
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                       loc=3,
                       ncol=len(unique_labels),
                       mode="expand",
                       borderaxespad=0.)

        if save_figure is True:
            plt.savefig(figname, bbox_inches='tight')

        if show_figure is True:
            plt.show()

class TSNE(Embedding):
    """
    Object wrapping ``sklearn`` implementation of t-distributed Stochastic Neighbour Embedding (t-SNE) that also includes useful plotting functions

    For description of attributes, please see the ``sklearn`` documentation on TSNE.
    """
    def __init__(self, n_components=2,  perplexity=30, early_exaggeration=4.0, learning_rate=100, n_iter=1000, n_iter_without_progress=30, min_grad_norm=1e-7, metric='precomputed', init='random', verbose=0, random_state=None, method='barnes_hut', angle=0.5):
        self.algorithm = tsne(n_components=n_components,
                              perplexity=perplexity,
                              early_exaggeration=early_exaggeration,
                              learning_rate=learning_rate,
                              n_iter=n_iter,
                              n_iter_without_progress=n_iter_without_progress,
                              min_grad_norm=min_grad_norm,
                              metric=metric,
                              init=init,
                              verbose=verbose,
                              random_state=random_state,
                              method=method,
                              angle=angle)
