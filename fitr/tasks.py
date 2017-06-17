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
Module containing code to implement tasks with simulated subjects

References
----------
.. [Daw2011] Daw, N.D. et al. (2011) Model-based influences on humans’ choices and striatal prediction errors. Neuron 69, 1204–1215

Module Documentation
--------------------
"""
import warnings
import numpy as np
from .models.synthetic_data import SyntheticData

# Warn user of the deprecated modules
warnings.warn(
    "The generative_models module is deprecated, and will be removed in future releases. Please use similar capacities task-specific modules",
    DeprecationWarning
)

#=========================================================================
#
#   SIMPLE BANDIT TASK
#
#=========================================================================

class bandit(object):
    """
    Simple one-step bandit task.

    Attributes
    ----------
    narms : int
        Number of arms
    rewards : ndarray(shape=(2))
        First entry is the reward, if gained, and the second entry is the magnitude of the loss
    rprob : {ndarray(shape=(narms)), 'stochastic'}
        Probabilty of reward for each arm of the task. One can either specify the probabilities for each arm or enter 'stochastic,' which will vary the reward probability by a gaussian random walk

    Methods
    -------
    simulate(nsubjects,ntrials)
        Runs the task on simulated subjects

    """

    def __init__(self, narms=2, rewards=[1, 0], rprob='stochastic', rprob_sd=0.025, rprob_bounds=[0.2, 0.8]):
        self.narms = narms
        self.rewards = rewards
        self.rprob_sd = rprob_sd

        if rprob == 'stochastic':
            self.rprob_bounds = rprob_bounds
        else:
            if len(rprob) != self.narms:
                print('Reward probability vector must have one entry per arm')
                return
            if np.any(rprob > 1) or np.any(rprob < 0):
                print('Reward probabilities must all lie between 0 and 1')
                return

            self.rprob = rprob

    def simulate(self, ntrials, params):
        """
        Simulates the task

        Parameters
        ----------
        ntrials : int > 0
            Number of trials to run
        params : ndarray(shape=(nsubjects X nparams))
            Parameters for each subject

        Returns
        -------
        SyntheticData

        """

        nsubjects = np.shape(params)[0]

        # Initialize reward paths
        path_max = self.rprob_bounds[0]
        path_min = self.rprob_bounds[1]
        path_sd = self.rprob_sd

        results = SyntheticData()
        results.params = params

        for i in range(0, nsubjects):
            paths = np.random.uniform(
                path_min, path_max, size=[ntrials + 1, 2])

            # initialize subject-level value table
            Q = np.zeros(2)
            lr = params[i, 0]
            cr = params[i, 1]

            results.data[i] = {'S': np.zeros(ntrials),
                               'A': np.zeros(ntrials),
                               'R': np.zeros(ntrials),
                               'RPE': np.zeros(ntrials)}

            for t in range(0, ntrials):
                a = action(cr * Q)
                r = reward(a, paths[t, :])

                # learn
                rpe = r - Q[a]
                Q[a] = Q[a] + lr * rpe

                # store values
                results.data[i]['S'][t] = 0
                results.data[i]['A'][t] = a
                results.data[i]['R'][t] = r
                results.data[i]['RPE'][t] = rpe

                # update reward probabilities
                rand_step = np.random.normal(0, path_sd, size=2)
                paths[
                    t + 1, :] = np.maximum(np.minimum(paths[t, :] + rand_step, path_max), path_min)

        # Convert standard data format into MCMC formatting
        actions = np.zeros([ntrials, nsubjects])
        rewards = np.zeros([ntrials, nsubjects])
        for i in range(nsubjects):
            actions[:, i] = results.data[i]['A'] + 1
            rewards[:, i] = results.data[i]['R']

        results.data_mcmc = {
            'N': nsubjects,
            'T': ntrials,
            'A': actions.astype(int),
            'R': rewards
        }

        return results

#=========================================================================
#
#   UTILITY FUNCTIONS
#
#=========================================================================

def action(x):
    """
    Selects an action based on state-action values

    Parameters
    ----------
    x : ndarray
        Array of action values (scaled by inverse softmax temperature).

    Returns
    -------
    int
        The index corresponding to the selected action

    Notes
    -----
    This function computes the softmax probability for each action in the input array, and subsequently samples from a multinomial distribution parameterized by the results of the softmax computation. Finally, it returns the index where the value is equal to 1 (i.e. which action was selected).

    """
    p = np.exp(x) / np.sum(np.exp(x))
    return np.argmax(np.random.multinomial(1, pvals=p))


def reward(a, paths):
    """
    Samples from a probability distribution over rewards

    Parameters
    ----------
    a : int
        The action that was selected at the time step
    paths : ndarray(size=(n_actions))
        Current reward probability for each action

    Returns
    -------
    int (1 or 0)
        Reward

    Notes
    -----
    This function samples from a binomial distribution with n=1 (i.e.  Bernoulli distribution) parameterized by the probability of reward for the currently selected action.

    .. math:: Binomial(n=1, p=P(reward|a_t))

    """
    return np.random.binomial(1, p=paths[a])
