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
import matplotlib.pyplot as plt

def heatmap(X, xlab=None, ylab=None, title=None, ticks=False, show_figure=True, save_figure=False, figsize=None, figname='heat.pdf'):
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
    xticks : bool (default True)
        Whether to show x-ticks/xticklabels

    show_figure : bool
        Whether to show the figure
    save_figure : bool
        Whether to save the figure
    figsize : (optional) list
        Figure dimensions

    """
    if figsize is None:
        figsize = [5, 5]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(X)

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

    if show_figure is True:
        plt.show()
