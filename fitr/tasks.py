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
#   ORTHOGONAL GO-NOGO
#
#===============================================================================

#class ortho_gng(object):
#    """
#    Model of the orthogonalized go-nogo task from Guitart-Masip et al. (2012)
#
#    Attributes
#    ----------
#    rewards : list
#
#    References
#    ----------
#    [1] Guitart-Masip, M. et al. (2012) Go and no-go learning in reward and punishment: Interactions between affect and effect. Neuroimage 62, 154–166
#    """
#
#    def __init__(self, rewards=[1, 0, -1]):
#        self.rewards=rewards
#
#    def simulate(self, ntrials, params):
#        nsubjects = np.shape(params)[0]
#        pass


#===============================================================================
#
#   TWO-STEP TASK
#
#=========================================================================


class twostep(object):
    """
    Model of the two-step task (Daw et al. 2011).

    Attributes
    ----------
    ptrans : ndarray
        Probability of transitioning from state 0 to either state 1 or 2 depending on the choice made at the first step of the task.

    Methods
    -------
    simulate(self, ntrials, params)
        Generates synthetic data from the task.

    References
    ----------
    [1] Daw, N.D. et al. (2011) Model-based influences on humans’ choices and striatal prediction errors. Neuron 69, 1204–1215
    """
    def __init__(self, ptrans=0.7, rewards=[1, 0]):
        """
        Instantiates a likelihood function object for the two-step task

        Parameters
        ----------
        ptrans : float between 0 and 1
            The high transition probability
        rewards : 2 X 1 list or ndarray
            The reward and non-reward magnitudes, respectively

        """
        self.ptrans = np.array([1 - ptrans, ptrans])

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

        results = SyntheticData()
        results.params = params

        # initialize reward paths
        path_max = 0.8
        path_min = 0.2
        path_sd = 0.025

        for i in range(nsubjects):
            # Initialize paths within subjects
            paths = np.random.uniform(
                path_min, path_max, size=[ntrials + 1, 4])

            lr = params[i, 0]
            cr = params[i, 1]
            w = params[i, 2]

            results.data[i] = {
                'S': np.zeros([ntrials, 2]),
                'A': np.zeros([ntrials, 2]),
                'R': np.zeros(ntrials)
            }

            Q = np.zeros([3, 2])
            Qmb = np.zeros([3, 2])
            Qmf = np.zeros([3, 2])

            for t in range(ntrials):
                s1 = int(0)
                a1 = int(action(cr * Q[s1, :]))

                s2 = int(np.random.binomial(1, p=self.ptrans[a1]) + 1)
                a2 = int(action(cr * Q[s2, :]))

                rprob = paths[t, :]
                rprob = np.reshape(rprob, (2, 2))
                r = np.random.binomial(1, p=rprob[s2 - 1, a2])

                # Update model-free values
                Qmf[s2, a2] = Qmf[s2, a2] + lr * (r - Qmf[s2, a2])
                Qmf[s1, a1] = Qmf[s1, a1] + lr * (Qmf[s2, a2] - Qmf[s1, a1])

                # Update model based values
                Qmb[0, a1] = self.ptrans[1] * \
                    np.max(Qmf[1, :]) + self.ptrans[0] * np.max(Qmf[2, :])
                Qmb[0, a2] = self.ptrans[1] * \
                    np.max(Qmf[2, :]) + self.ptrans[0] * np.max(Qmf[1, :])

                # Linear combination of MF and MB
                Q = w * Qmb + (1 - w) * Qmf

                # Store data
                results.data[i]['S'][t, 0] = s1
                results.data[i]['S'][t, 1] = s2
                results.data[i]['A'][t, 0] = a1
                results.data[i]['A'][t, 1] = a2
                results.data[i]['R'][t] = r

                # Update reward probabilities
                rand_step = np.random.normal(0, path_sd, size=4)
                paths[
                    t + 1, :] = np.maximum(np.minimum(paths[t, :] + rand_step, path_max), path_min)

        # Convert standard data format into MCMC formatting
        step1_action = np.zeros([ntrials, nsubjects])
        step2_state  = np.zeros([ntrials, nsubjects])
        step2_action = np.zeros([ntrials, nsubjects])
        rewards = np.zeros([ntrials, nsubjects])
        for i in range(nsubjects):
            step1_action[:, i] = results.data[i]['A'][:, 0] + 1
            step2_state[:, i]  = results.data[i]['S'][:, 1] + 1
            step2_action[:, i] = results.data[i]['A'][:, 1] + 1
            rewards[:, i] = results.data[i]['R']

        results.data_mcmc = {
            'N': nsubjects,
            'T': ntrials,
            'S2': step2_state.astype(int),
            'A1': step1_action.astype(int),
            'A2': step2_action.astype(int),
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
