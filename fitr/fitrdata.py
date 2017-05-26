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
Module containing data objects used in fitr.

Module Documentation
--------------------
"""

import numpy as np
import matplotlib.pyplot as plt

class SyntheticData(object):
    """
    Object representing synthetic data

    Attributes
    ----------
    data : dict
        Dictionary containing data formatted for fitr's model fitting tools (except MCMC via Stan)
    data_mcmc : dict
        Dictionary containing task data formatted for use with MCMC via Stan
    params : ndarray(shape=(nsubjects X nparams))
        Subject parameters

    Methods
    -------
    cumreward_param_plot(self, alpha=0.9)
        Plots the cumulative reward against model parameters. Useful to determine the relationship between reward acquisition and model parameters for a given task.
    plot_cumreward(self)
        Plots the cumulative reward over time for each subject
    """

    def __init__(self):
        self.data = {}
        self.data_mcmc = {}
        self.params = None
        self.paramnames = None

    def cumreward_param_plot(self, alpha=0.9):
        if self.params is not None:
            nsubjects = np.shape(self.params)[0]
            creward = np.zeros(nsubjects)
            for i in range(0, nsubjects):
                creward[i] = np.sum(self.data[i]['R'])

            nparams = np.shape(self.params)[1]
            fig, ax = plt.subplots(1, nparams, figsize=(15, 5))
            for i in range(0, nparams):
                ax[i].scatter(self.params[:, i], creward, c='k', alpha=alpha)
                ax[i].set_xlabel(self.paramnames[i])
                ax[i].set_ylabel('Total Reward')

            plt.suptitle('Cumulative Reward vs. Parameters')
            plt.show()
        else:
            print('ERROR: There are no parameters assigned')
            return

    def plot_cumreward(self):
        """ Plots cumulative reward over time for each subject"""
        nsubjects = len(self.data)
        fig, ax = plt.subplots(1, 1)
        for i in range(nsubjects):
            nsteps = len(self.data[i]['R'])
            ax.plot(np.arange(nsteps), np.cumsum(self.data[i]['R']))

        ax.set_title('Cumulative Reward by Subject\n')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Reward')
        plt.show()
