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
Module containing code to implement simulations of a four armed bandit task with drifting reward probabilities.

Module Documentation
--------------------
"""

import numpy as np
from .synthetic_data import SyntheticData
from .taskmodel import TaskModel

from ..rlparams import LearningRate
from ..rlparams import ChoiceRandomness
from ..rlparams import RewardSensitivity
from ..rlparams import Perseveration
from ..utils import action as _action
from ..utils import logsumexp

class lr_cr(TaskModel):
    """
    Learner with only learning rate and choice randomness parameters.

    Attributes
    ----------
    LR : fitr.rlparams.LearningRate object
        Learning rate object
    CR : fitr.rlparams.ChoiceRandomness object
        Choice randomness object
    gm : GenerativeModel
    ptrans : float in range (0, 1)
        The dominant transition probability (default = 0.7).

    Methods
    -------
    loglikelihood()
        Loglikelihood function for parameter estimation using optimization
    set_generativemodel()
        Sets the Stan code for the model
    simulate()
        Simulates data for a group of artificial subjects

    """
    def __init__(self, narms=4, LR=LearningRate(), CR=ChoiceRandomness()):

        self.LR = LR
        self.CR = CR

        # Set generative model
        self.set_gm(path='stancode/driftbandit/lrcr.stan',
                    paramnames_long=['Learning Rate',
                                     'Choice Randomness'],
                    paramnames_code=['lr', 'cr'])

        # Task parameters
        self.narms = narms

    def loglikelihood(self, params, states, actions, rewards):
        """
        Likelihood function for parameter estimation

        Parameters
        ----------
        params : ndarray(shape=2)
            Current parameter estimates for learning rate and choice randomness.
        states : ndarray(shape=ntrials)
            Subject-level states
        actions : ndarray(shape=ntrials)
            Subject-level actions
        rewards : ndarray(shape=ntrials)
            Subject-level rewards

        Returns
        -------
        float

        """
        lr = params[0]
        cr = params[1]

        ntrials = len(actions)
        Q = np.zeros([1, self.narms])
        loglik = 0

        for t in range(0, ntrials):
            s = int(states[t])
            a = int(actions[t])
            r = rewards[t]

            loglik = loglik + cr*Q[s, a] - logsumexp(cr*Q)
            Q[s, a] = Q[s, a] + lr*(r - Q[s, a])

        return loglik

    def simulate(self, ntrials, nsubjects, group_id=None, preset_rpaths=None, rpath_max=0.75, rpath_min=0.25, rpath_sd=0.025, rpath_common=False):
        """
        Simulates the task from a group

        Parameters
        ----------
        ntrials : int > 0
            Number of trials to run
        nsubjects : int > 0
            Number of subjects to simulate
        group_id : int (default=None)
            Identifier for the group of simulated subjects
        preset_rpaths : ndarray(shape=(ntrials, narms, nsubjects))
            Array of preset reward paths. If rpath_common is True, then this can be an (ntrials X 4) array.
        rpath_max : float on interval (0, 1)
            Maximum probability of reward
        rpath_min : float on interval (0, 1)
            Minimum probability of reward
        rpath_sd  : float on interval (0, +Inf)
            Standard deviation of the Gaussian random walk for reward probabilities
        rpath_common : bool
            Whether the reward paths for step-2 choices should be the same across all subjects

        Returns
        -------
        SyntheticData
        """

        # Generate group of subjects
        params = np.zeros([nsubjects, 2])
        params[:, 0] = self.LR.sample(size=nsubjects)
        params[:, 1] = self.CR.sample(size=nsubjects)

        results = SyntheticData()
        results.params = params
        results.paramnames = [self.LR.name,
                              self.CR.name]

        # Set reward paths
        if preset_rpaths is None:
            if rpath_common is True:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, self.narms])

                for t in range(ntrials):
                    rand_step = np.random.normal(0, rpath_sd, size=self.narms)
                    paths[t+1, :] = np.maximum(np.minimum(paths[t, :] + rand_step, rpath_max), rpath_min)

            else:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, self.narms, nsubjects])

                for i in range(nsubjects):
                    for t in range(ntrials):
                        rand_step = np.random.normal(0, rpath_sd, size=self.narms)
                        paths[t+1, :, i] = np.maximum(np.minimum(paths[t, :, i] + rand_step, rpath_max), rpath_min)

        else:
            paths = preset_rpaths

        for i in range(nsubjects):
            # Set subject-level reward path
            if rpath_common is True and preset_rpaths is None:
                subj_rpath = paths
            else:
                subj_rpath = paths[:, :, i]

            lr = params[i, 0]
            cr = params[i, 1]

            results.data[i] = {
                'S' : np.zeros(ntrials),
                'A' : np.zeros(ntrials),
                'R' : np.zeros(ntrials),
                'RPE' : np.zeros(ntrials),
                'Q' : np.zeros([ntrials, self.narms])
            }

            if group_id is not None:
                results.data[i]['G'] = group_id

            Q = np.zeros([1, self.narms])

            for t in range(ntrials):
                s = int(0)
                a = int(_action(cr * Q[s, :]))
                r = np.random.binomial(1, p=subj_rpath[t, a])

                # Update model-free values
                rpe = (r - Q[s, a])
                Q[s, a] = Q[s, a] + lr * rpe

                # Store data
                results.data[i]['S'][t] = s
                results.data[i]['A'][t] = a
                results.data[i]['R'][t] = r
                results.data[i]['RPE'][t] = rpe
                results.data[i]['Q'][t, :] = Q


        # Convert standard data format into MCMC formatting
        actions = np.zeros([ntrials, nsubjects])
        rewards = np.zeros([ntrials, nsubjects])
        for i in range(nsubjects):
            actions[:, i] = results.data[i]['A'] + 1
            rewards[:, i] = results.data[i]['R']

        results.data_mcmc = {
            'K': self.narms,
            'N': nsubjects,
            'T': ntrials,
            'A': actions.astype(int),
            'R': rewards
        }

        if group_id is not None:
            results.data_mcmc['G'] = np.array([group_id]*nsubjects)

        return results

class lr_cr_rs(TaskModel):
    """
    Learner with learning rate, choice randomness, and reward sensitivity parameters.

    Attributes
    ----------
    LR : fitr.rlparams.LearningRate
        Learning rate object
    CR : fitr.rlparams.ChoiceRandomness
        Choice randomness object
    RS : fitr.rlparams.RewardSensitivity
        Reward sensitivity parameter object
    gm : GenerativeModel
    ptrans : float in range (0, 1)
        The dominant transition probability (default = 0.7).

    Methods
    -------
    loglikelihood()
        Loglikelihood function for parameter estimation using optimization
    simulate()
        Simulates data for a group of artificial subjects

    """
    def __init__(self, narms=4, LR=LearningRate(), CR=ChoiceRandomness(), RS=RewardSensitivity()):

        self.LR = LR
        self.CR = CR
        self.RS = RS

        # Set generative model
        self.set_gm(path='stancode/driftbandit/lrcrrs.stan',
                    paramnames_long=['Learning Rate',
                                     'Choice Randomness',
                                     'RewardSensitivity'],
                    paramnames_code=['lr', 'cr', 'rs'])

        # Task parameters
        self.narms = narms

    def loglikelihood(self, params, states, actions, rewards):
        """
        Likelihood function for parameter estimation

        Parameters
        ----------
        params : ndarray(shape=2)
            Current parameter estimates for learning rate and choice randomness.
        states : ndarray(shape=ntrials)
            Subject-level states
        actions : ndarray(shape=ntrials)
            Subject-level actions
        rewards : ndarray(shape=ntrials)
            Subject-level rewards

        Returns
        -------
        float

        """
        lr = params[0]
        cr = params[1]
        rs = params[2]

        ntrials = len(actions)
        Q = np.zeros([1, self.narms])
        loglik = 0

        for t in range(0, ntrials):
            s = int(states[t])
            a = int(actions[t])
            r = rewards[t]

            loglik = loglik + cr*Q[s, a] - logsumexp(cr*Q)
            Q[s, a] = Q[s, a] + lr*(rs*r - Q[s, a])

        return loglik

    def simulate(self, ntrials, nsubjects, group_id=None, preset_rpaths=None, rpath_max=0.75, rpath_min=0.25, rpath_sd=0.025, rpath_common=False):
        """
        Simulates the task from a group

        Parameters
        ----------
        ntrials : int > 0
            Number of trials to run
        nsubjects : int > 0
            Number of subjects to simulate
        group_id : int (default=None)
            Identifier for the group of simulated subjects
        preset_rpaths : ndarray(shape=(ntrials, self.narms, nsubjects))
            Array of preset reward paths. If rpath_common is True, then this can be an (ntrials X 4) array.
        rpath_max : float on interval (0, 1)
            Maximum probability of reward
        rpath_min : float on interval (0, 1)
            Minimum probability of reward
        rpath_sd  : float on interval (0, +Inf)
            Standard deviation of the Gaussian random walk for reward probabilities
        rpath_common : bool
            Whether the reward paths for step-2 choices should be the same across all subjects

        Returns
        -------
        SyntheticData
        """

        # Generate group of subjects
        params = np.zeros([nsubjects, 3])
        params[:, 0] = self.LR.sample(size=nsubjects)
        params[:, 1] = self.CR.sample(size=nsubjects)
        params[:, 2] = self.RS.sample(size=nsubjects)

        results = SyntheticData()
        results.params = params
        results.paramnames = [self.LR.name,
                              self.CR.name,
                              self.RS.name]

        # Set reward paths
        if preset_rpaths is None:
            if rpath_common is True:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, self.narms])

                for t in range(ntrials):
                    rand_step = np.random.normal(0, rpath_sd, size=self.narms)
                    paths[t+1, :] = np.maximum(np.minimum(paths[t, :] + rand_step, rpath_max), rpath_min)

            else:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, self.narms, nsubjects])

                for i in range(nsubjects):
                    for t in range(ntrials):
                        rand_step = np.random.normal(0, rpath_sd, size=self.narms)
                        paths[t+1, :, i] = np.maximum(np.minimum(paths[t, :, i] + rand_step, rpath_max), rpath_min)

        else:
            paths = preset_rpaths

        for i in range(nsubjects):
            # Set subject-level reward path
            if rpath_common is True and preset_rpaths is None:
                subj_rpath = paths
            else:
                subj_rpath = paths[:, :, i]

            lr = params[i, 0]
            cr = params[i, 1]
            rs = params[i, 2]

            results.data[i] = {
                'S' : np.zeros(ntrials),
                'A' : np.zeros(ntrials),
                'R' : np.zeros(ntrials),
                'RPE' : np.zeros(ntrials),
                'Q' : np.zeros([ntrials, self.narms])
            }

            if group_id is not None:
                results.data[i]['G'] = group_id

            Q = np.zeros([1, self.narms])

            for t in range(ntrials):
                s = int(0)
                a = int(_action(cr * Q[s, :]))
                r = np.random.binomial(1, p=subj_rpath[t, a])

                # Update model-free values
                rpe = (rs*r - Q[s, a])
                Q[s, a] = Q[s, a] + lr * rpe

                # Store data
                results.data[i]['S'][t] = s
                results.data[i]['A'][t] = a
                results.data[i]['R'][t] = r
                results.data[i]['RPE'][t] = rpe
                results.data[i]['Q'][t, :] = Q


        # Convert standard data format into MCMC formatting
        actions = np.zeros([ntrials, nsubjects])
        rewards = np.zeros([ntrials, nsubjects])
        for i in range(nsubjects):
            actions[:, i] = results.data[i]['A'] + 1
            rewards[:, i] = results.data[i]['R']

        results.data_mcmc = {
            'K': self.narms,
            'N': nsubjects,
            'T': ntrials,
            'A': actions.astype(int),
            'R': rewards
        }

        if group_id is not None:
            results.data_mcmc['G'] = np.array([group_id]*nsubjects)

        return results

class lr_cr_p(TaskModel):
    """
    Learner with learning rate, choice randomness, and perseveration parameters.

    Attributes
    ----------
    LR : fitr.rlparams.LearningRate
        Learning rate object
    CR : fitr.rlparams.ChoiceRandomness
        Choice randomness object
    P : fitr.rlparams.Perseveration
        Perseveration parameter object
    gm : GenerativeModel
    ptrans : float in range (0, 1)
        The dominant transition probability (default = 0.7).

    Methods
    -------
    loglikelihood()
        Loglikelihood function for parameter estimation using optimization
    set_generativemodel()
        Sets the Stan code for the model
    simulate()
        Simulates data for a group of artificial subjects

    """
    def __init__(self, narms=4, LR=LearningRate(), CR=ChoiceRandomness(), P=Perseveration()):

        self.LR = LR
        self.CR = CR
        self.P = P

        # Set generative model
        self.set_gm(path='stancode/driftbandit/lrcrp.stan',
                    paramnames_long=['Learning Rate',
                                     'Choice Randomness',
                                     'Perseveration'],
                    paramnames_code=['lr', 'cr', 'persev'])

        # Task parameters
        self.narms = narms

    def loglikelihood(self, params, states, actions, rewards):
        """
        Likelihood function for parameter estimation

        Parameters
        ----------
        params : ndarray(shape=2)
            Current parameter estimates for learning rate and choice randomness.
        states : ndarray(shape=ntrials)
            Subject-level states
        actions : ndarray(shape=ntrials)
            Subject-level actions
        rewards : ndarray(shape=ntrials)
            Subject-level rewards

        Returns
        -------
        float

        """
        lr = params[0]
        cr = params[1]
        persev = params[2]

        ntrials = len(actions)
        Q = np.zeros([1, self.narms])
        loglik = 0

        # Initialize perseveration vector
        p_vec = np.zeros(self.narms)
        a_last = 100
        for t in range(0, ntrials):
            s = int(states[t])
            a = int(actions[t])
            r = rewards[t]

            loglik = loglik + cr*Q[s, a] - logsumexp(cr*Q)
            Q[s, a] = Q[s, a] + lr*(r - Q[s, a])

            # Adjust Q for perseveration
            if a == a_last:
                p_vec[a] = 1
            else:
                p_vec = np.zeros(self.narms)

            Q[s,:] = Q[s,:] + persev*p_vec

            a_last = a

        return loglik

    def simulate(self, ntrials, nsubjects, group_id=None, preset_rpaths=None, rpath_max=0.75, rpath_min=0.25, rpath_sd=0.025, rpath_common=False):
        """
        Simulates the task from a group

        Parameters
        ----------
        ntrials : int > 0
            Number of trials to run
        nsubjects : int > 0
            Number of subjects to simulate
        group_id : int (default=None)
            Identifier for the group of simulated subjects
        preset_rpaths : ndarray(shape=(ntrials, narms, nsubjects))
            Array of preset reward paths. If rpath_common is True, then this can be an (ntrials X 4) array.
        rpath_max : float on interval (0, 1)
            Maximum probability of reward
        rpath_min : float on interval (0, 1)
            Minimum probability of reward
        rpath_sd  : float on interval (0, +Inf)
            Standard deviation of the Gaussian random walk for reward probabilities
        rpath_common : bool
            Whether the reward paths for step-2 choices should be the same across all subjects

        Returns
        -------
        SyntheticData
        """

        # Generate group of subjects
        params = np.zeros([nsubjects, 4])
        params[:, 0] = self.LR.sample(size=nsubjects)
        params[:, 1] = self.CR.sample(size=nsubjects)
        params[:, 2] = self.P.sample(size=nsubjects)

        results = SyntheticData()
        results.params = params
        results.paramnames = [self.LR.name,
                              self.CR.name,
                              self.P.name]

        # Set reward paths
        if preset_rpaths is None:
            if rpath_common is True:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, self.narms])

                for t in range(ntrials):
                    rand_step = np.random.normal(0, rpath_sd, size=self.narms)
                    paths[t+1, :] = np.maximum(np.minimum(paths[t, :] + rand_step, rpath_max), rpath_min)

            else:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, self.narms, nsubjects])

                for i in range(nsubjects):
                    for t in range(ntrials):
                        rand_step = np.random.normal(0, rpath_sd, size=self.narms)
                        paths[t+1, :, i] = np.maximum(np.minimum(paths[t, :, i] + rand_step, rpath_max), rpath_min)

        else:
            paths = preset_rpaths

        for i in range(nsubjects):
            # Set subject-level reward path
            if rpath_common is True and preset_rpaths is None:
                subj_rpath = paths
            else:
                subj_rpath = paths[:, :, i]

            lr = params[i, 0]
            cr = params[i, 1]
            persev = params[i, 2]

            results.data[i] = {
                'S' : np.zeros(ntrials),
                'A' : np.zeros(ntrials),
                'R' : np.zeros(ntrials),
                'RPE' : np.zeros(ntrials),
                'Q' : np.zeros([ntrials, self.narms])
            }

            if group_id is not None:
                results.data[i]['G'] = group_id

            Q = np.zeros([1, self.narms])

            # Initialize perseveration vector
            p_vec = np.zeros(self.narms)
            a_last = 100

            for t in range(ntrials):
                s = int(0)
                a = int(_action(cr * Q[s, :]))
                r = np.random.binomial(1, p=subj_rpath[t, a])

                # Update model-free values
                rpe = (r - Q[s, a])
                Q[s, a] = Q[s, a] + lr * rpe

                # Adjust Q for perseveration
                if a == a_last:
                    p_vec[a] = 1
                else:
                    p_vec = np.zeros(self.narms)

                Q[s,:] = Q[s,:] + persev*p_vec

                a_last = a

                # Store data
                results.data[i]['S'][t] = s
                results.data[i]['A'][t] = a
                results.data[i]['R'][t] = r
                results.data[i]['RPE'][t] = rpe
                results.data[i]['Q'][t, :] = Q


        # Convert standard data format into MCMC formatting
        actions = np.zeros([ntrials, nsubjects])
        rewards = np.zeros([ntrials, nsubjects])
        for i in range(nsubjects):
            actions[:, i] = results.data[i]['A'] + 1
            rewards[:, i] = results.data[i]['R']

        results.data_mcmc = {
            'K': self.narms,
            'N': nsubjects,
            'T': ntrials,
            'A': actions.astype(int),
            'R': rewards
        }

        if group_id is not None:
            results.data_mcmc['G'] = np.array([group_id]*nsubjects)

        return results

class lr_cr_rs_p(TaskModel):
    """
    Learner with learning rate, choice randomness, reward sensitivity, and perseveration parameters.

    Attributes
    ----------
    LR : fitr.rlparams.LearningRate
        Learning rate object
    CR : fitr.rlparams.ChoiceRandomness
        Choice randomness object
    RS : fitr.rlparams.RewardSensitivity
        Reward sensitivity parameter object
    P : fitr.rlparams.Perseveration
        Perseveration parameter object
    gm : GenerativeModel
    ptrans : float in range (0, 1)
        The dominant transition probability (default = 0.7).

    Methods
    -------
    loglikelihood()
        Loglikelihood function for parameter estimation using optimization
    set_generativemodel()
        Sets the Stan code for the model
    simulate()
        Simulates data for a group of artificial subjects

    """
    def __init__(self, narms=4, LR=LearningRate(), CR=ChoiceRandomness(), RS=RewardSensitivity(), P=Perseveration()):

        self.LR = LR
        self.CR = CR
        self.RS = RS
        self.P = P

        # Set generative model
        self.set_gm(path='stancode/driftbandit/lrcrrsp.stan',
                    paramnames_long=['Learning Rate',
                                     'Choice Randomness',
                                     'Reward Sensitivity',
                                     'Perseveration'],
                    paramnames_code=['lr', 'cr', 'rs', 'persev'])

        # Task parameters
        self.narms = narms

    def loglikelihood(self, params, states, actions, rewards):
        """
        Likelihood function for parameter estimation

        Parameters
        ----------
        params : ndarray(shape=2)
            Current parameter estimates for learning rate and choice randomness.
        states : ndarray(shape=ntrials)
            Subject-level states
        actions : ndarray(shape=ntrials)
            Subject-level actions
        rewards : ndarray(shape=ntrials)
            Subject-level rewards

        Returns
        -------
        float

        """
        lr = params[0]
        cr = params[1]
        rs = params[2]
        persev = params[3]

        ntrials = len(actions)
        Q = np.zeros([1, self.narms])
        loglik = 0

        # Initialize perseveration vector
        p_vec = np.zeros(self.narms)
        a_last = 100
        for t in range(0, ntrials):
            s = int(states[t])
            a = int(actions[t])
            r = rewards[t]

            loglik = loglik + cr*Q[s, a] - logsumexp(cr*Q)
            Q[s, a] = Q[s, a] + lr*(rs*r - Q[s, a])

            # Adjust Q for perseveration
            if a == a_last:
                p_vec[a] = 1
            else:
                p_vec = np.zeros(self.narms)

            Q[s,:] = Q[s,:] + persev*p_vec

            a_last = a

        return loglik

    def simulate(self, ntrials, nsubjects, group_id=None, preset_rpaths=None, rpath_max=0.75, rpath_min=0.25, rpath_sd=0.025, rpath_common=False):
        """
        Simulates the task from a group

        Parameters
        ----------
        ntrials : int > 0
            Number of trials to run
        nsubjects : int > 0
            Number of subjects to simulate
        group_id : int (default=None)
            Identifier for the group of simulated subjects
        preset_rpaths : ndarray(shape=(ntrials, narms, nsubjects))
            Array of preset reward paths. If rpath_common is True, then this can be an (ntrials X 4) array.
        rpath_max : float on interval (0, 1)
            Maximum probability of reward
        rpath_min : float on interval (0, 1)
            Minimum probability of reward
        rpath_sd  : float on interval (0, +Inf)
            Standard deviation of the Gaussian random walk for reward probabilities
        rpath_common : bool
            Whether the reward paths for step-2 choices should be the same across all subjects

        Returns
        -------
        SyntheticData
        """

        # Generate group of subjects
        params = np.zeros([nsubjects, 4])
        params[:, 0] = self.LR.sample(size=nsubjects)
        params[:, 1] = self.CR.sample(size=nsubjects)
        params[:, 2] = self.RS.sample(size=nsubjects)
        params[:, 3] = self.P.sample(size=nsubjects)

        results = SyntheticData()
        results.params = params
        results.paramnames = [self.LR.name,
                              self.CR.name,
                              self.RS.name,
                              self.P.name]

        # Set reward paths
        if preset_rpaths is None:
            if rpath_common is True:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, self.narms])

                for t in range(ntrials):
                    rand_step = np.random.normal(0, rpath_sd, size=self.narms)
                    paths[t+1, :] = np.maximum(np.minimum(paths[t, :] + rand_step, rpath_max), rpath_min)

            else:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, self.narms, nsubjects])

                for i in range(nsubjects):
                    for t in range(ntrials):
                        rand_step = np.random.normal(0, rpath_sd, size=self.narms)
                        paths[t+1, :, i] = np.maximum(np.minimum(paths[t, :, i] + rand_step, rpath_max), rpath_min)

        else:
            paths = preset_rpaths

        for i in range(nsubjects):
            # Set subject-level reward path
            if rpath_common is True and preset_rpaths is None:
                subj_rpath = paths
            else:
                subj_rpath = paths[:, :, i]

            lr = params[i, 0]
            cr = params[i, 1]
            rs = params[i, 2]
            persev = params[i, 3]

            results.data[i] = {
                'S' : np.zeros(ntrials),
                'A' : np.zeros(ntrials),
                'R' : np.zeros(ntrials),
                'RPE' : np.zeros(ntrials),
                'Q' : np.zeros([ntrials, self.narms])
            }

            if group_id is not None:
                results.data[i]['G'] = group_id

            Q = np.zeros([1, self.narms])

            # Initialize perseveration vector
            p_vec = np.zeros(self.narms)
            a_last = 100

            for t in range(ntrials):
                s = int(0)
                a = int(_action(cr * Q[s, :]))
                r = np.random.binomial(1, p=subj_rpath[t, a])

                # Update model-free values
                rpe = (rs*r - Q[s, a])
                Q[s, a] = Q[s, a] + lr * rpe

                # Adjust Q for perseveration
                if a == a_last:
                    p_vec[a] = 1
                else:
                    p_vec = np.zeros(self.narms)

                Q[s,:] = Q[s,:] + persev*p_vec

                a_last = a

                # Store data
                results.data[i]['S'][t] = s
                results.data[i]['A'][t] = a
                results.data[i]['R'][t] = r
                results.data[i]['RPE'][t] = rpe
                results.data[i]['Q'][t, :] = Q


        # Convert standard data format into MCMC formatting
        actions = np.zeros([ntrials, nsubjects])
        rewards = np.zeros([ntrials, nsubjects])
        for i in range(nsubjects):
            actions[:, i] = results.data[i]['A'] + 1
            rewards[:, i] = results.data[i]['R']

        results.data_mcmc = {
            'K': self.narms,
            'N': nsubjects,
            'T': ntrials,
            'A': actions.astype(int),
            'R': rewards
        }

        if group_id is not None:
            results.data_mcmc['G'] = np.array([group_id]*nsubjects)

        return results

class dummy(TaskModel):
    """
    Dummy learner with choice randomness.

    Attributes
    ----------
    CR : fitr.rlparams.ChoiceRandomness
        Choice randomness object
    ptrans : float in range (0, 1)
        The dominant transition probability (default = 0.7).

    Methods
    -------
    loglikelihood()
        Loglikelihood function for parameter estimation using optimization
    set_generativemodel()
        Sets the Stan code for the model
    simulate()
        Simulates data for a group of artificial subjects

    """
    def __init__(self, narms=4, CR=ChoiceRandomness()):

        self.CR = CR

        # Task parameters
        self.narms = narms

    def loglikelihood(self, params, states, actions, rewards):
        """
        Likelihood function for parameter estimation

        Parameters
        ----------
        params : ndarray(shape=2)
            Current parameter estimates for learning rate and choice randomness.
        states : ndarray(shape=ntrials)
            Subject-level states
        actions : ndarray(shape=ntrials)
            Subject-level actions
        rewards : ndarray(shape=ntrials)
            Subject-level rewards

        Returns
        -------
        float

        """
        cr  = params[0]

        ntrials = len(actions)
        Q = np.zeros([1, self.narms])
        loglik = 0

        for t in range(0, ntrials):
            s = int(states[t])
            a = int(actions[t])
            loglik = loglik + cr*Q[s, a] - logsumexp(cr*Q)

        return loglik

    def set_generativemodel(self):
        pass

    def simulate(self, ntrials, nsubjects, group_id=None, preset_rpaths=None, rpath_max=0.75, rpath_min=0.25, rpath_sd=0.025, rpath_common=False):
        """
        Simulates the task from a group

        Parameters
        ----------
        ntrials : int > 0
            Number of trials to run
        nsubjects : int > 0
            Number of subjects to simulate
        group_id : int (default=None)
            Identifier for the group of simulated subjects
        preset_rpaths : ndarray(shape=(ntrials, narms, nsubjects))
            Array of preset reward paths. If rpath_common is True, then this can be an (ntrials X 4) array.
        rpath_max : float on interval (0, 1)
            Maximum probability of reward
        rpath_min : float on interval (0, 1)
            Minimum probability of reward
        rpath_sd  : float on interval (0, +Inf)
            Standard deviation of the Gaussian random walk for reward probabilities
        rpath_common : bool
            Whether the reward paths for step-2 choices should be the same across all subjects

        Returns
        -------
        SyntheticData
        """

        # Generate group of subjects
        params = np.zeros([nsubjects, 1])
        params[:, 0] = self.CR.sample(size=nsubjects)

        results = SyntheticData()
        results.params = params
        results.paramnames = [self.CR.name]

        # Set reward paths
        if preset_rpaths is None:
            if rpath_common is True:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, self.narms])

                for t in range(ntrials):
                    rand_step = np.random.normal(0, rpath_sd, size=self.narms)
                    paths[t+1, :] = np.maximum(np.minimum(paths[t, :] + rand_step, rpath_max), rpath_min)

            else:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, self.narms, nsubjects])

                for i in range(nsubjects):
                    for t in range(ntrials):
                        rand_step = np.random.normal(0, rpath_sd, size=self.narms)
                        paths[t+1, :, i] = np.maximum(np.minimum(paths[t, :, i] + rand_step, rpath_max), rpath_min)

        else:
            paths = preset_rpaths

        for i in range(nsubjects):
            # Set subject-level reward path
            if rpath_common is True and preset_rpaths is None:
                subj_rpath = paths
            else:
                subj_rpath = paths[:, :, i]

            cr = params[i, 0]

            results.data[i] = {
                'S' : np.zeros(ntrials),
                'A' : np.zeros(ntrials),
                'R' : np.zeros(ntrials),
                'RPE' : np.zeros(ntrials),
                'Q' : np.zeros([ntrials, self.narms])
            }

            if group_id is not None:
                results.data[i]['G'] = group_id

            Q = np.zeros([1, self.narms])

            for t in range(ntrials):
                s = int(0)
                a = int(_action(cr * Q[s, :]))
                r = np.random.binomial(1, p=subj_rpath[t, a])

                # Update model-free values
                rpe = (r - Q[s, a])

                # Store data
                results.data[i]['S'][t] = s
                results.data[i]['A'][t] = a
                results.data[i]['R'][t] = r
                results.data[i]['RPE'][t] = rpe
                results.data[i]['Q'][t, :] = Q


        # Convert standard data format into MCMC formatting
        actions = np.zeros([ntrials, nsubjects])
        rewards = np.zeros([ntrials, nsubjects])
        for i in range(nsubjects):
            actions[:, i] = results.data[i]['A'] + 1
            rewards[:, i] = results.data[i]['R']

        results.data_mcmc = {
            'K': self.narms,
            'N': nsubjects,
            'T': ntrials,
            'A': actions.astype(int),
            'R': rewards
        }

        if group_id is not None:
            results.data_mcmc['G'] = np.array([group_id]*nsubjects)

        return results
