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
Core plotting functions

Module Documentation
--------------------
"""
import itertools
import matplotlib.pyplot as plt

def heatmap(X, xlab=None, ylab=None, title=None, ticks=False, interpolation='none', save_figure=False, figsize=None, figname='heat.pdf'):
    """
    Plots a heatmap based on an input matrix

    Parameters
    ----------
    xlab : (optional) str
        x-axis label
    ylab : (optional) str
        y-axis label
    title : (optional) str
        Plot title
    ticks : bool (default True)
        Whether to show ticks and ticklabels
    interpolation : str
        Matplotlib interpolation method for image
    save_figure : bool
        Whether to save the figure
    figsize : (optional) list
        Figure dimensions

    """
    if figsize is None:
        figsize = [5, 5]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(X, interpolation=interpolation)

    if title is not None:
        ax.set_title(title)
    if xlab is not None:
        ax.set_title(xlab)
    if ylab is not None:
        ax.set_title(ylab)

    if ticks is False:
        ax.tick_params(axis='both',
                       left='off',
                       top='off',
                       right='off',
                       bottom='off',
                       labelleft='off',
                       labeltop='off',
                       labelright='off',
                       labelbottom='off')

    if save_figure is True:
        plt.savefig(figname, bbox_inches="tight")

    return ax

def confusion_matrix(X,
                     classes,
                     normalize=False,
                     round_digits = 2,
                     title='Confusion matrix',
                     xlabel='Predicted Label',
                     ylabel='True Label',
                     file_dir=None,
                     filename=None,
                     cmap=plt.cm.Blues):
    """
    Plots a heatmap/confusion matrix according to a matrix, while printing the numbers inside the grid cells.

    Parameters
    ----------
    X : ndarray
        The matrix to be plotted
    normalize : bool
        Whether the matrix entries should be normalized
    round_digits : int >= 0
        The number of digits to which the matrix entries should be rounded
    title : str
        Plot title
    xlabel : str
    ylabel : str
    file_dir : str
        The directory to which the file should be saved
    filename : str
    cmap : matplotlib colourmap

    Returns
    -------
    matplotlib.figure

    """
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    if figsize is None:
        figsize = (8, 8)

    fig = plt.figure(figsize=figsize)
    plt.imshow(X, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        X = np.round(X.astype('float') / X.sum(axis=1)[:, np.newaxis], round_digits)

    thresh = X.max() / 2.
    for i, j in itertools.product(range(X.shape[0]), range(X.shape[1])):
        plt.text(j, i, X[i, j],
                 horizontalalignment="center",
                 color="white" if X[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    if filename is not None:
        plt.savefig(file_dir + filename, bbox_inches='tight')

    return fig
