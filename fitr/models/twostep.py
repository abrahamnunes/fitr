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
Module containing code to implement simulations of Two-Step Task and fit models to behavioural data from that task.

References
----------
.. [Daw2011] Daw, N.D. et al. (2011) Model-based influences on humans’ choices and striatal prediction errors. Neuron 69, 1204–1215

Module Documentation
--------------------
"""
import numpy as np
from .synthetic_data import SyntheticData

from ..rlparams import LearningRate
from ..rlparams import ChoiceRandomness
from ..rlparams import EligibilityTrace
from ..rlparams import RewardSensitivity
from ..rlparams import Perseveration
from ..utils import action, logsumexp

class lr_cr_mf(object):
    """
    Two-step task model of a model free learner with only learning rate and choice randomness parameters.

    Attributes
    ----------
    LR : fitr.rlparams.LearningRate object
        Learning rate object
    CR : fitr.rlparams.ChoiceRandomness object
        Choice randomness object
    generative_model : str
        Stan code for fitr.MCMC
    ptrans : float in range (0, 1)
        The dominant transition probability (default = 0.7).

    Methods
    -------
    loglikelihood()
        Loglikelihood function for parameter estimation using optimization
    set_generativemodel()
        Sets the Stan code for the model
    make_group()
        Creates a group of subjects
    simulate()
        Simulates data for a group of artificial subjects

    """
    def __init__(self, LR=LearningRate(), CR=ChoiceRandomness(), ptrans=0.7):

        self.LR = LR
        self.CR = CR
        self.generative_model = self.set_generativemodel()

        # Task parameters
        self.ptrans = np.array([1 - ptrans, ptrans])

    def loglikelihood(self, params, states, actions, rewards):
        """
        Likelihood function for parameter estimation

        Parameters
        ----------
        params : ndarray(shape=2)
            Current parameter estimates for learning rate and choice randomness.
        states : ndarray(shape=[ntrials, 2])
            Subject-level states
        actions : ndarray(shape=[ntrials, 2])
            Subject-level actions
        rewards : ndarray(shape=ntrials)
            Subject-level rewards

        Returns
        -------
        float

        """
        # Initialize parameters
        lr = params[0]
        cr = params[1]

        # Initialize log-likelihood
        loglik = 0

        # Initialize Q arrays
        Q   = np.zeros([3, 2])

        ntrials = np.shape(states)[0]
        for t in range(ntrials):
            s1 = int(states[t,0])
            s2 = int(states[t,1])
            a1 = int(actions[t,0])
            a2 = int(actions[t,1])
            r = rewards[t]

            loglik = loglik + cr*Q[s1,a1] - logsumexp(cr*Q[s1,:])
            loglik = loglik + cr*Q[s2,a2] - logsumexp(cr*Q[s2,:])

            # Learning
            rpe = r - Q[s2, a2]
            Q[s1, a1] = Q[s1, a1] + lr*(Q[s2, a2]-Q[s1, a1])
            Q[s2, a2] = Q[s2, a2] + lr*rpe
            Q[s1, a1] = Q[s1, a1] + lr*rpe

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
        preset_rpaths : ndarray(shape=(ntrials, 4, nsubjects))
            Array of preset reward paths. If rpath_common is True, then this can be an (ntrials X 4) array.
        rpath_max : float on interval (0, 1)
            Maximum probability of reward at a given step-2 choice
        rpath_min : float on interval (0, 1)
            Minimum probability of reward at a given step-2 choice
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
                                          size=[ntrials + 1, 4])

                for t in range(ntrials):
                    rand_step = np.random.normal(0, rpath_sd, size=4)
                    paths[t+1, :] = np.maximum(np.minimum(paths[t, :] + rand_step, rpath_max), rpath_min)

            else:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, 4, nsubjects])

                for i in range(nsubjects):
                    for t in range(ntrials):
                        rand_step = np.random.normal(0, rpath_sd, size=4)
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
                'S' : np.zeros([ntrials, 2]),
                'A' : np.zeros([ntrials, 2]),
                'R' : np.zeros(ntrials),
                'RPE' : np.zeros(ntrials),
                'Q' : np.zeros([3, 2, ntrials])
            }

            if group_id is not None:
                results.data[i]['G'] = group_id

            Q = np.zeros([3, 2])

            for t in range(ntrials):
                s1 = int(0)
                a1 = int(action(cr * Q[s1, :]))

                s2 = int(np.random.binomial(1, p=self.ptrans[a1]) + 1)
                a2 = int(action(cr * Q[s2, :]))

                rprob = subj_rpath[t, :]
                rprob = np.reshape(rprob, (2, 2))
                r = np.random.binomial(1, p=rprob[s2 - 1, a2])

                # Update model-free values
                rpe = (r - Q[s2, a2])
                Q[s1, a1] = Q[s1, a1] + lr * (Q[s2, a2] - Q[s1, a1])
                Q[s2, a2] = Q[s2, a2] + lr * rpe
                Q[s1, a1] = Q[s1, a1] + lr * rpe

                # Store data
                results.data[i]['S'][t, 0] = s1
                results.data[i]['S'][t, 1] = s2
                results.data[i]['A'][t, 0] = a1
                results.data[i]['A'][t, 1] = a2
                results.data[i]['R'][t] = r
                results.data[i]['RPE'][t] = rpe
                results.data[i]['Q'][:, :, t] = Q


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

        if group_id is not None:
            results.data_mcmc['G'] = np.array([group_id]*nsubjects)

        return results

class lr_cr_rs_mf(object):
    """
    Two-step task model of a model free learner with learning rate, choice randomness, and reward sensitivity parameters.

    Attributes
    ----------
    LR : fitr.rlparams.LearningRate object
    CR : fitr.rlparams.ChoiceRandomness object
    RS : fitr.rlparams.RewardSensitivity object
    generative_model : str
        Stan code for fitr.MCMC
    ptrans : float in range (0, 1)
        The dominant transition probability (default = 0.7).

    Methods
    -------
    loglikelihood()
        Loglikelihood function for parameter estimation using optimization
    set_generativemodel()
        Sets the Stan code for the model
    make_group()
        Creates a group of subjects
    simulate()
        Simulates data for a group of artificial subjects

    """
    def __init__(self, LR=LearningRate(), CR=ChoiceRandomness(), RS=RewardSensitivity(), ptrans=0.7):

        self.LR = LR
        self.CR = CR
        self.RS = RS
        self.generative_model = self.set_generativemodel()

        # Task parameters
        self.ptrans = np.array([1 - ptrans, ptrans])

    def loglikelihood(self, params, states, actions, rewards):
        """
        Likelihood function for parameter estimation

        Parameters
        ----------
        params : ndarray(shape=2)
            Current parameter estimates for learning rate and choice randomness.
        states : ndarray(shape=[ntrials, 2])
            Subject-level states
        actions : ndarray(shape=[ntrials, 2])
            Subject-level actions
        rewards : ndarray(shape=ntrials)
            Subject-level rewards

        Returns
        -------
        float

        """
        # Initialize parameters
        lr = params[0]
        cr = params[1]
        rs = params[2]

        # Initialize log-likelihood
        loglik = 0

        # Initialize Q arrays
        Q   = np.zeros([3, 2])

        ntrials = np.shape(states)[0]

        for t in range(ntrials):
            s1 = int(states[t,0])
            s2 = int(states[t,1])
            a1 = int(actions[t,0])
            a2 = int(actions[t,1])
            r = rewards[t]

            loglik = loglik + cr*Q[s1,a1] - logsumexp(cr*Q[s1,:])
            loglik = loglik + cr*Q[s2,a2] - logsumexp(cr*Q[s2,:])

            # Learning
            rpe = rs*r - Q[s2, a2]
            Q[s1, a1] = Q[s1, a1] + lr*(Q[s2, a2]-Q[s1, a1])
            Q[s2, a2] = Q[s2, a2] + lr*rpe
            Q[s1, a1] = Q[s1, a1] + lr*rpe

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
        preset_rpaths : ndarray(shape=(ntrials, 4, nsubjects))
            Array of preset reward paths. If rpath_common is True, then this can be an (ntrials X 4) array.
        rpath_max : float on interval (0, 1)
            Maximum probability of reward at a given step-2 choice
        rpath_min : float on interval (0, 1)
            Minimum probability of reward at a given step-2 choice
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
                                          size=[ntrials + 1, 4])

                for t in range(ntrials):
                    rand_step = np.random.normal(0, rpath_sd, size=4)
                    paths[t+1, :] = np.maximum(np.minimum(paths[t, :] + rand_step, rpath_max), rpath_min)

            else:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, 4, nsubjects])

                for i in range(nsubjects):
                    for t in range(ntrials):
                        rand_step = np.random.normal(0, rpath_sd, size=4)
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
                'S' : np.zeros([ntrials, 2]),
                'A' : np.zeros([ntrials, 2]),
                'R' : np.zeros(ntrials),
                'RPE' : np.zeros(ntrials),
                'Q' : np.zeros([3, 2, ntrials])
            }

            if group_id is not None:
                results.data[i]['G'] = group_id

            Q = np.zeros([3, 2])

            for t in range(ntrials):
                s1 = int(0)
                a1 = int(action(cr * Q[s1, :]))

                s2 = int(np.random.binomial(1, p=self.ptrans[a1]) + 1)
                a2 = int(action(cr * Q[s2, :]))

                rprob = subj_rpath[t, :]
                rprob = np.reshape(rprob, (2, 2))
                r = np.random.binomial(1, p=rprob[s2 - 1, a2])

                # Update model-free values
                rpe = (rs*r - Q[s2, a2])
                Q[s1, a1] = Q[s1, a1] + lr * (Q[s2, a2] - Q[s1, a1])
                Q[s2, a2] = Q[s2, a2] + lr*rpe
                Q[s1, a1] = Q[s1, a1] + lr*rpe

                # Store data
                results.data[i]['S'][t, 0] = s1
                results.data[i]['S'][t, 1] = s2
                results.data[i]['A'][t, 0] = a1
                results.data[i]['A'][t, 1] = a2
                results.data[i]['R'][t] = r
                results.data[i]['RPE'][t] = rpe
                results.data[i]['Q'][:, :, t] = Q


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

        if group_id is not None:
            results.data_mcmc['G'] = np.array([group_id]*nsubjects)

        return results

class lr_cr_et_mf(object):
    """
    Two-step task model of a model free learner with learning rate, choice randomness, and eligibility trace parameters.

    Attributes
    ----------
    LR : fitr.rlparams.LearningRate object
        Learning rate object
    CR : fitr.rlparams.ChoiceRandomness object
        Choice randomness object
    ET : fitr.rlparams.EligibilityTrace object
        Eligibility trace object
    generative_model : str
        Stan code for fitr.MCMC
    ptrans : float in range (0, 1)
        The dominant transition probability (default = 0.7).

    Methods
    -------
    loglikelihood()
        Loglikelihood function for parameter estimation using optimization
    set_generativemodel()
        Sets the Stan code for the model
    make_group()
        Creates a group of subjects
    simulate()
        Simulates data for a group of artificial subjects

    """
    def __init__(self, LR=LearningRate(), CR=ChoiceRandomness(), ET=EligibilityTrace(), ptrans=0.7):

        self.LR = LR
        self.CR = CR
        self.ET = ET
        self.generative_model = self.set_generativemodel()

        # Task parameters
        self.ptrans = np.array([1 - ptrans, ptrans])

    def loglikelihood(self, params, states, actions, rewards):
        """
        Likelihood function for parameter estimation

        Parameters
        ----------
        params : ndarray(shape=2)
            Current parameter estimates for learning rate and choice randomness.
        states : ndarray(shape=[ntrials, 2])
            Subject-level states
        actions : ndarray(shape=[ntrials, 2])
            Subject-level actions
        rewards : ndarray(shape=ntrials)
            Subject-level rewards

        Returns
        -------
        float

        """
        # Initialize parameters
        lr = params[0]
        cr = params[1]
        et = params[2]

        # Initialize log-likelihood
        loglik = 0

        # Initialize Q arrays
        Q   = np.zeros([3, 2])

        ntrials = np.shape(states)[0]
        for t in range(ntrials):
            s1 = int(states[t,0])
            s2 = int(states[t,1])
            a1 = int(actions[t,0])
            a2 = int(actions[t,1])
            r = rewards[t]

            loglik = loglik + cr*Q[s1,a1] - logsumexp(cr*Q[s1,:])
            loglik = loglik + cr*Q[s2,a2] - logsumexp(cr*Q[s2,:])

            # Learning
            rpe = r - Q[s2, a2]
            Q[s1, a1] = Q[s1, a1] + lr*(Q[s2, a2]-Q[s1, a1])
            Q[s2, a2] = Q[s2, a2] + lr*rpe
            Q[s1, a1] = Q[s1, a1] + lr*rpe*et

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
        preset_rpaths : ndarray(shape=(ntrials, 4, nsubjects))
            Array of preset reward paths. If rpath_common is True, then this can be an (ntrials X 4) array.
        rpath_max : float on interval (0, 1)
            Maximum probability of reward at a given step-2 choice
        rpath_min : float on interval (0, 1)
            Minimum probability of reward at a given step-2 choice
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
        params[:, 2] = self.ET.sample(size=nsubjects)

        results = SyntheticData()
        results.params = params
        results.paramnames = [self.LR.name,
                              self.CR.name,
                              self.ET.name]

        # Set reward paths
        if preset_rpaths is None:
            if rpath_common is True:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, 4])

                for t in range(ntrials):
                    rand_step = np.random.normal(0, rpath_sd, size=4)
                    paths[t+1, :] = np.maximum(np.minimum(paths[t, :] + rand_step, rpath_max), rpath_min)

            else:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, 4, nsubjects])

                for i in range(nsubjects):
                    for t in range(ntrials):
                        rand_step = np.random.normal(0, rpath_sd, size=4)
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
            et = params[i, 2]

            results.data[i] = {
                'S' : np.zeros([ntrials, 2]),
                'A' : np.zeros([ntrials, 2]),
                'R' : np.zeros(ntrials),
                'RPE' : np.zeros(ntrials),
                'Q' : np.zeros([3, 2, ntrials])
            }

            if group_id is not None:
                results.data[i]['G'] = group_id

            Q = np.zeros([3, 2])

            for t in range(ntrials):
                s1 = int(0)
                a1 = int(action(cr * Q[s1, :]))

                s2 = int(np.random.binomial(1, p=self.ptrans[a1]) + 1)
                a2 = int(action(cr * Q[s2, :]))

                rprob = subj_rpath[t, :]
                rprob = np.reshape(rprob, (2, 2))
                r = np.random.binomial(1, p=rprob[s2 - 1, a2])

                # Update model-free values
                rpe = (r - Q[s2, a2])
                Q[s1, a1] = Q[s1, a1] + lr * (Q[s2, a2] - Q[s1, a1])
                Q[s2, a2] = Q[s2, a2] + lr * rpe
                Q[s1, a1] = Q[s1, a1] + lr*rpe*et

                # Store data
                results.data[i]['S'][t, 0] = s1
                results.data[i]['S'][t, 1] = s2
                results.data[i]['A'][t, 0] = a1
                results.data[i]['A'][t, 1] = a2
                results.data[i]['R'][t] = r
                results.data[i]['RPE'][t] = rpe
                results.data[i]['Q'][:, :, t] = Q


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

        if group_id is not None:
            results.data_mcmc['G'] = np.array([group_id]*nsubjects)

        return results

class lr_cr_p_mf(object):
    """
    Two-step task model of a model free learner with learning rate, choice randomness, and perseveration parameters.

    Attributes
    ----------
    LR : fitr.rlparams.LearningRate object
        Learning rate object
    CR : fitr.rlparams.ChoiceRandomness object
        Choice randomness object
    P : fitr.rlparams.Perseveration object
        Perseveration parameter object
    generative_model : str
        Stan code for fitr.MCMC
    ptrans : float in range (0, 1)
        The dominant transition probability (default = 0.7).

    Methods
    -------
    loglikelihood()
        Loglikelihood function for parameter estimation using optimization
    set_generativemodel()
        Sets the Stan code for the model
    make_group()
        Creates a group of subjects
    simulate()
        Simulates data for a group of artificial subjects

    """
    def __init__(self, LR=LearningRate(), CR=ChoiceRandomness(), P=Perseveration(), ptrans=0.7):

        self.LR = LR
        self.CR = CR
        self.P = P
        self.generative_model = self.set_generativemodel()

        # Task parameters
        self.ptrans = np.array([1 - ptrans, ptrans])

    def loglikelihood(self, params, states, actions, rewards):
        """
        Likelihood function for parameter estimation

        Parameters
        ----------
        params : ndarray(shape=2)
            Current parameter estimates for learning rate and choice randomness.
        states : ndarray(shape=[ntrials, 2])
            Subject-level states
        actions : ndarray(shape=[ntrials, 2])
            Subject-level actions
        rewards : ndarray(shape=ntrials)
            Subject-level rewards

        Returns
        -------
        float

        """
        # Initialize parameters
        lr = params[0]
        cr = params[1]
        persev = params[2]

        # Initialize log-likelihood
        loglik = 0

        # Initialize Q arrays
        Q   = np.zeros([3, 2])

        # Initialize perseveration vector
        p_vec = np.zeros(2)

        ntrials = np.shape(states)[0]
        a_last = 100
        for t in range(ntrials):
            s1 = int(states[t,0])
            s2 = int(states[t,1])
            a1 = int(actions[t,0])
            a2 = int(actions[t,1])
            r = rewards[t]

            loglik = loglik + cr*Q[s1,a1] - logsumexp(cr*Q[s1,:])
            loglik = loglik + cr*Q[s2,a2] - logsumexp(cr*Q[s2,:])

            # Learning
            rpe = r - Q[s2, a2]
            Q[s1, a1] = Q[s1, a1] + lr*(Q[s2, a2]-Q[s1, a1])
            Q[s2, a2] = Q[s2, a2] + lr*rpe
            Q[s1, a1] = Q[s1, a1] + lr*rpe

            # Adjust Q for perseveration
            if a1 == a_last:
                p_vec[a1] = 1
            else:
                p_vec = np.zeros(2)

            Q[s1,:] = Q[s1,:] + persev*p_vec

            a_last = a1

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
        preset_rpaths : ndarray(shape=(ntrials, 4, nsubjects))
            Array of preset reward paths. If rpath_common is True, then this can be an (ntrials X 4) array.
        rpath_max : float on interval (0, 1)
            Maximum probability of reward at a given step-2 choice
        rpath_min : float on interval (0, 1)
            Minimum probability of reward at a given step-2 choice
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
                                          size=[ntrials + 1, 4])

                for t in range(ntrials):
                    rand_step = np.random.normal(0, rpath_sd, size=4)
                    paths[t+1, :] = np.maximum(np.minimum(paths[t, :] + rand_step, rpath_max), rpath_min)

            else:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, 4, nsubjects])

                for i in range(nsubjects):
                    for t in range(ntrials):
                        rand_step = np.random.normal(0, rpath_sd, size=4)
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
                'S' : np.zeros([ntrials, 2]),
                'A' : np.zeros([ntrials, 2]),
                'R' : np.zeros(ntrials),
                'RPE' : np.zeros(ntrials),
                'Q' : np.zeros([3, 2, ntrials])
            }

            if group_id is not None:
                results.data[i]['G'] = group_id

            Q = np.zeros([3, 2])

            # Initialize perseveration vector
            p_vec = np.zeros(2)
            a_last = 100

            for t in range(ntrials):
                s1 = int(0)
                a1 = int(action(cr * Q[s1, :]))

                s2 = int(np.random.binomial(1, p=self.ptrans[a1]) + 1)
                a2 = int(action(cr * Q[s2, :]))

                rprob = subj_rpath[t, :]
                rprob = np.reshape(rprob, (2, 2))
                r = np.random.binomial(1, p=rprob[s2 - 1, a2])

                # Update model-free values
                rpe = (r - Q[s2, a2])
                Q[s1, a1] = Q[s1, a1] + lr * (Q[s2, a2] - Q[s1, a1])
                Q[s2, a2] = Q[s2, a2] + lr * rpe
                Q[s1, a1] = Q[s1, a1] + lr*rpe

                # Adjust Q for perseveration
                if a1 == a_last:
                    p_vec[a1] = 1
                else:
                    p_vec = np.zeros(2)

                Q[s1,:] = Q[s1,:] + persev*p_vec

                a_last = a1

                # Store data
                results.data[i]['S'][t, 0] = s1
                results.data[i]['S'][t, 1] = s2
                results.data[i]['A'][t, 0] = a1
                results.data[i]['A'][t, 1] = a2
                results.data[i]['R'][t] = r
                results.data[i]['RPE'][t] = rpe
                results.data[i]['Q'][:, :, t] = Q


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

        if group_id is not None:
            results.data_mcmc['G'] = np.array([group_id]*nsubjects)

        return results

class lr_cr_et_p_mf(object):
    """
    Two-step task model of a model free learner with learning rate, choice randomness, eligibility trace, and perseveration parameters.

    Attributes
    ----------
    LR : fitr.rlparams.LearningRate object
        Learning rate object
    CR : fitr.rlparams.ChoiceRandomness object
        Choice randomness object
    ET : fitr.rlparams.EligibilityTrace object
        Eligibility trace object
    P : fitr.rlparams.Perseveration object
        Perseveration parameter object
    generative_model : str
        Stan code for fitr.MCMC
    ptrans : float in range (0, 1)
        The dominant transition probability (default = 0.7).

    Methods
    -------
    loglikelihood()
        Loglikelihood function for parameter estimation using optimization
    set_generativemodel()
        Sets the Stan code for the model
    make_group()
        Creates a group of subjects
    simulate()
        Simulates data for a group of artificial subjects

    """
    def __init__(self, LR=LearningRate(), CR=ChoiceRandomness(), ET=EligibilityTrace(), P=Perseveration(), ptrans=0.7):

        self.LR = LR
        self.CR = CR
        self.ET = ET
        self.P = P
        self.generative_model = self.set_generativemodel()

        # Task parameters
        self.ptrans = np.array([1 - ptrans, ptrans])

    def loglikelihood(self, params, states, actions, rewards):
        """
        Likelihood function for parameter estimation

        Parameters
        ----------
        params : ndarray(shape=2)
            Current parameter estimates for learning rate and choice randomness.
        states : ndarray(shape=[ntrials, 2])
            Subject-level states
        actions : ndarray(shape=[ntrials, 2])
            Subject-level actions
        rewards : ndarray(shape=ntrials)
            Subject-level rewards

        Returns
        -------
        float

        """
        # Initialize parameters
        lr = params[0]
        cr = params[1]
        et = params[2]
        persev = params[3]

        # Initialize log-likelihood
        loglik = 0

        # Initialize Q arrays
        Q   = np.zeros([3, 2])

        # Initialize perseveration vector
        p_vec = np.zeros(2)

        ntrials = np.shape(states)[0]
        a_last = 100
        for t in range(ntrials):
            s1 = int(states[t,0])
            s2 = int(states[t,1])
            a1 = int(actions[t,0])
            a2 = int(actions[t,1])
            r = rewards[t]

            loglik = loglik + cr*Q[s1,a1] - logsumexp(cr*Q[s1,:])
            loglik = loglik + cr*Q[s2,a2] - logsumexp(cr*Q[s2,:])

            # Learning
            rpe = r - Q[s2, a2]
            Q[s1, a1] = Q[s1, a1] + lr*(Q[s2, a2]-Q[s1, a1])
            Q[s2, a2] = Q[s2, a2] + lr*rpe
            Q[s1, a1] = Q[s1, a1] + lr*rpe*et

            # Adjust Q for perseveration
            if a1 == a_last:
                p_vec[a1] = 1
            else:
                p_vec = np.zeros(2)

            Q[s1,:] = Q[s1,:] + persev*p_vec

            a_last = a1

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
        preset_rpaths : ndarray(shape=(ntrials, 4, nsubjects))
            Array of preset reward paths. If rpath_common is True, then this can be an (ntrials X 4) array.
        rpath_max : float on interval (0, 1)
            Maximum probability of reward at a given step-2 choice
        rpath_min : float on interval (0, 1)
            Minimum probability of reward at a given step-2 choice
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
        params[:, 2] = self.ET.sample(size=nsubjects)
        params[:, 3] = self.P.sample(size=nsubjects)

        results = SyntheticData()
        results.params = params
        results.paramnames = [self.LR.name,
                              self.CR.name,
                              self.ET.name,
                              self.P.name]

        # Set reward paths
        if preset_rpaths is None:
            if rpath_common is True:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, 4])

                for t in range(ntrials):
                    rand_step = np.random.normal(0, rpath_sd, size=4)
                    paths[t+1, :] = np.maximum(np.minimum(paths[t, :] + rand_step, rpath_max), rpath_min)

            else:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, 4, nsubjects])

                for i in range(nsubjects):
                    for t in range(ntrials):
                        rand_step = np.random.normal(0, rpath_sd, size=4)
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
            et = params[i, 2]
            persev = params[i, 3]

            results.data[i] = {
                'S' : np.zeros([ntrials, 2]),
                'A' : np.zeros([ntrials, 2]),
                'R' : np.zeros(ntrials),
                'RPE' : np.zeros(ntrials),
                'Q' : np.zeros([3, 2, ntrials])
            }

            if group_id is not None:
                results.data[i]['G'] = group_id

            Q = np.zeros([3, 2])

            # Initialize perseveration vector
            p_vec = np.zeros(2)
            a_last = 100

            for t in range(ntrials):
                s1 = int(0)
                a1 = int(action(cr * Q[s1, :]))

                s2 = int(np.random.binomial(1, p=self.ptrans[a1]) + 1)
                a2 = int(action(cr * Q[s2, :]))

                rprob = subj_rpath[t, :]
                rprob = np.reshape(rprob, (2, 2))
                r = np.random.binomial(1, p=rprob[s2 - 1, a2])

                # Update model-free values
                rpe = (r - Q[s2, a2])
                Q[s1, a1] = Q[s1, a1] + lr * (Q[s2, a2] - Q[s1, a1])
                Q[s2, a2] = Q[s2, a2] + lr * rpe
                Q[s1, a1] = Q[s1, a1] + lr*rpe*et

                # Adjust Q for perseveration
                if a1 == a_last:
                    p_vec[a1] = 1
                else:
                    p_vec = np.zeros(2)

                Q[s1,:] = Q[s1,:] + persev*p_vec

                a_last = a1

                # Store data
                results.data[i]['S'][t, 0] = s1
                results.data[i]['S'][t, 1] = s2
                results.data[i]['A'][t, 0] = a1
                results.data[i]['A'][t, 1] = a2
                results.data[i]['R'][t] = r
                results.data[i]['RPE'][t] = rpe
                results.data[i]['Q'][:, :, t] = Q


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

        if group_id is not None:
            results.data_mcmc['G'] = np.array([group_id]*nsubjects)

        return results

class lr_cr_rs_p_mf(object):
    """
    Two-step task model of a model free learner with learning rate, choice randomness, reward sensitivity, and perseveration parameters.

    Attributes
    ----------
    LR : fitr.rlparams.LearningRate object
    CR : fitr.rlparams.ChoiceRandomness object
    RS : fitr.rlparams.RewardSensitivity object
    P : fitr.rlparams.Perseveration object
    generative_model : str
        Stan code for fitr.MCMC
    ptrans : float in range (0, 1)
        The dominant transition probability (default = 0.7).

    Methods
    -------
    loglikelihood()
        Loglikelihood function for parameter estimation using optimization
    set_generativemodel()
        Sets the Stan code for the model
    make_group()
        Creates a group of subjects
    simulate()
        Simulates data for a group of artificial subjects

    """
    def __init__(self, LR=LearningRate(), CR=ChoiceRandomness(), RS=RewardSensitivity(), P=Perseveration(), ptrans=0.7):

        self.LR = LR
        self.CR = CR
        self.RS = RS
        self.P = P
        self.generative_model = self.set_generativemodel()

        # Task parameters
        self.ptrans = np.array([1 - ptrans, ptrans])

    def loglikelihood(self, params, states, actions, rewards):
        """
        Likelihood function for parameter estimation

        Parameters
        ----------
        params : ndarray(shape=2)
            Current parameter estimates for learning rate and choice randomness.
        states : ndarray(shape=[ntrials, 2])
            Subject-level states
        actions : ndarray(shape=[ntrials, 2])
            Subject-level actions
        rewards : ndarray(shape=ntrials)
            Subject-level rewards

        Returns
        -------
        float

        """
        # Initialize parameters
        lr = params[0]
        cr = params[1]
        rs = params[2]
        persev = params[3]

        # Initialize log-likelihood
        loglik = 0

        # Initialize Q arrays
        Q   = np.zeros([3, 2])

        # Initialize perseveration vector
        p_vec = np.zeros(2)

        ntrials = np.shape(states)[0]
        a_last = 100
        for t in range(ntrials):
            s1 = int(states[t,0])
            s2 = int(states[t,1])
            a1 = int(actions[t,0])
            a2 = int(actions[t,1])
            r = rewards[t]

            loglik = loglik + cr*Q[s1,a1] - logsumexp(cr*Q[s1,:])
            loglik = loglik + cr*Q[s2,a2] - logsumexp(cr*Q[s2,:])

            # Learning
            rpe = rs*r - Q[s2, a2]
            Q[s1, a1] = Q[s1, a1] + lr*(Q[s2, a2]-Q[s1, a1])
            Q[s2, a2] = Q[s2, a2] + lr*rpe
            Q[s1, a1] = Q[s1, a1] + lr*rpe

            # Adjust Q for perseveration
            if a1 == a_last:
                p_vec[a1] = 1
            else:
                p_vec = np.zeros(2)

            Q[s1,:] = Q[s1,:] + persev*p_vec

            a_last = a1

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
        preset_rpaths : ndarray(shape=(ntrials, 4, nsubjects))
            Array of preset reward paths. If rpath_common is True, then this can be an (ntrials X 4) array.
        rpath_max : float on interval (0, 1)
            Maximum probability of reward at a given step-2 choice
        rpath_min : float on interval (0, 1)
            Minimum probability of reward at a given step-2 choice
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
                                          size=[ntrials + 1, 4])

                for t in range(ntrials):
                    rand_step = np.random.normal(0, rpath_sd, size=4)
                    paths[t+1, :] = np.maximum(np.minimum(paths[t, :] + rand_step, rpath_max), rpath_min)

            else:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, 4, nsubjects])

                for i in range(nsubjects):
                    for t in range(ntrials):
                        rand_step = np.random.normal(0, rpath_sd, size=4)
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
                'S' : np.zeros([ntrials, 2]),
                'A' : np.zeros([ntrials, 2]),
                'R' : np.zeros(ntrials),
                'RPE' : np.zeros(ntrials),
                'Q' : np.zeros([3, 2, ntrials])
            }

            if group_id is not None:
                results.data[i]['G'] = group_id

            Q = np.zeros([3, 2])

            # Initialize perseveration vector
            p_vec = np.zeros(2)
            a_last = 100

            for t in range(ntrials):
                s1 = int(0)
                a1 = int(action(cr * Q[s1, :]))

                s2 = int(np.random.binomial(1, p=self.ptrans[a1]) + 1)
                a2 = int(action(cr * Q[s2, :]))

                rprob = subj_rpath[t, :]
                rprob = np.reshape(rprob, (2, 2))
                r = np.random.binomial(1, p=rprob[s2 - 1, a2])

                # Update model-free values
                rpe = (rs*r - Q[s2, a2])
                Q[s1, a1] = Q[s1, a1] + lr * (Q[s2, a2] - Q[s1, a1])
                Q[s2, a2] = Q[s2, a2] + lr*rpe
                Q[s1, a1] = Q[s1, a1] + lr*rpe

                # Adjust Q for perseveration
                if a1 == a_last:
                    p_vec[a1] = 1
                else:
                    p_vec = np.zeros(2)

                Q[s1,:] = Q[s1,:] + persev*p_vec

                a_last = a1

                # Store data
                results.data[i]['S'][t, 0] = s1
                results.data[i]['S'][t, 1] = s2
                results.data[i]['A'][t, 0] = a1
                results.data[i]['A'][t, 1] = a2
                results.data[i]['R'][t] = r
                results.data[i]['RPE'][t] = rpe
                results.data[i]['Q'][:, :, t] = Q


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

        if group_id is not None:
            results.data_mcmc['G'] = np.array([group_id]*nsubjects)

        return results

class lr_cr_rs_et_p_mf(object):
    """
    Two-step task model of a model free learner with learning rate, choice randomness, reward sensitivity, eligibility trace, and perseveration parameters.

    Attributes
    ----------
    LR : fitr.rlparams.LearningRate object
    CR : fitr.rlparams.ChoiceRandomness object
    RS : fitr.rlparams.RewardSensitivity object
    ET : fitr.rlparams.EligibilityTrace object
    P : fitr.rlparams.Perseveration object
    generative_model : str
        Stan code for fitr.MCMC
    ptrans : float in range (0, 1)
        The dominant transition probability (default = 0.7).

    Methods
    -------
    loglikelihood()
        Loglikelihood function for parameter estimation using optimization
    set_generativemodel()
        Sets the Stan code for the model
    make_group()
        Creates a group of subjects
    simulate()
        Simulates data for a group of artificial subjects

    """
    def __init__(self, LR=LearningRate(), CR=ChoiceRandomness(), RS=RewardSensitivity(),  ET=EligibilityTrace(), P=Perseveration(), ptrans=0.7):

        self.LR = LR
        self.CR = CR
        self.RS = RS
        self.ET = ET
        self.P = P
        self.generative_model = self.set_generativemodel()

        # Task parameters
        self.ptrans = np.array([1 - ptrans, ptrans])

    def loglikelihood(self, params, states, actions, rewards):
        """
        Likelihood function for parameter estimation

        Parameters
        ----------
        params : ndarray(shape=2)
            Current parameter estimates for learning rate and choice randomness.
        states : ndarray(shape=[ntrials, 2])
            Subject-level states
        actions : ndarray(shape=[ntrials, 2])
            Subject-level actions
        rewards : ndarray(shape=ntrials)
            Subject-level rewards

        Returns
        -------
        float

        """
        # Initialize parameters
        lr = params[0]
        cr = params[1]
        rs = params[2]
        et = params[3]
        persev = params[4]

        # Initialize log-likelihood
        loglik = 0

        # Initialize Q arrays
        Q   = np.zeros([3, 2])

        # Initialize perseveration vector
        p_vec = np.zeros(2)

        ntrials = np.shape(states)[0]
        a_last = 100
        for t in range(ntrials):
            s1 = int(states[t,0])
            s2 = int(states[t,1])
            a1 = int(actions[t,0])
            a2 = int(actions[t,1])
            r = rewards[t]

            loglik = loglik + cr*Q[s1,a1] - logsumexp(cr*Q[s1,:])
            loglik = loglik + cr*Q[s2,a2] - logsumexp(cr*Q[s2,:])

            # Learning
            rpe = rs*r - Q[s2, a2]
            Q[s1, a1] = Q[s1, a1] + lr*(Q[s2, a2]-Q[s1, a1])
            Q[s2, a2] = Q[s2, a2] + lr*rpe
            Q[s1, a1] = Q[s1, a1] + lr*rpe*et

            # Adjust Q for perseveration
            if a1 == a_last:
                p_vec[a1] = 1
            else:
                p_vec = np.zeros(2)

            Q[s1,:] = Q[s1,:] + persev*p_vec

            a_last = a1

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
        preset_rpaths : ndarray(shape=(ntrials, 4, nsubjects))
            Array of preset reward paths. If rpath_common is True, then this can be an (ntrials X 4) array.
        rpath_max : float on interval (0, 1)
            Maximum probability of reward at a given step-2 choice
        rpath_min : float on interval (0, 1)
            Minimum probability of reward at a given step-2 choice
        rpath_sd  : float on interval (0, +Inf)
            Standard deviation of the Gaussian random walk for reward probabilities
        rpath_common : bool
            Whether the reward paths for step-2 choices should be the same across all subjects

        Returns
        -------
        SyntheticData
        """

        # Generate group of subjects
        params = np.zeros([nsubjects, 5])
        params[:, 0] = self.LR.sample(size=nsubjects)
        params[:, 1] = self.CR.sample(size=nsubjects)
        params[:, 2] = self.RS.sample(size=nsubjects)
        params[:, 3] = self.ET.sample(size=nsubjects)
        params[:, 4] = self.P.sample(size=nsubjects)

        results = SyntheticData()
        results.params = params
        results.paramnames = [self.LR.name,
                              self.CR.name,
                              self.RS.name,
                              self.ET.name,
                              self.P.name]

        # Set reward paths
        if preset_rpaths is None:
            if rpath_common is True:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, 4])

                for t in range(ntrials):
                    rand_step = np.random.normal(0, rpath_sd, size=4)
                    paths[t+1, :] = np.maximum(np.minimum(paths[t, :] + rand_step, rpath_max), rpath_min)

            else:
                paths = np.random.uniform(rpath_min,
                                          rpath_max,
                                          size=[ntrials + 1, 4, nsubjects])

                for i in range(nsubjects):
                    for t in range(ntrials):
                        rand_step = np.random.normal(0, rpath_sd, size=4)
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
            et = params[i, 3]
            persev = params[i, 4]

            results.data[i] = {
                'S' : np.zeros([ntrials, 2]),
                'A' : np.zeros([ntrials, 2]),
                'R' : np.zeros(ntrials),
                'RPE' : np.zeros(ntrials),
                'Q' : np.zeros([3, 2, ntrials])
            }

            if group_id is not None:
                results.data[i]['G'] = group_id

            Q = np.zeros([3, 2])

            # Initialize perseveration vector
            p_vec = np.zeros(2)
            a_last = 100

            for t in range(ntrials):
                s1 = int(0)
                a1 = int(action(cr * Q[s1, :]))

                s2 = int(np.random.binomial(1, p=self.ptrans[a1]) + 1)
                a2 = int(action(cr * Q[s2, :]))

                rprob = subj_rpath[t, :]
                rprob = np.reshape(rprob, (2, 2))
                r = np.random.binomial(1, p=rprob[s2 - 1, a2])

                # Update model-free values
                rpe = (rs*r - Q[s2, a2])
                Q[s1, a1] = Q[s1, a1] + lr * (Q[s2, a2] - Q[s1, a1])
                Q[s2, a2] = Q[s2, a2] + lr * rpe
                Q[s1, a1] = Q[s1, a1] + lr*rpe*et

                # Adjust Q for perseveration
                if a1 == a_last:
                    p_vec[a1] = 1
                else:
                    p_vec = np.zeros(2)

                Q[s1,:] = Q[s1,:] + persev*p_vec

                a_last = a1

                # Store data
                results.data[i]['S'][t, 0] = s1
                results.data[i]['S'][t, 1] = s2
                results.data[i]['A'][t, 0] = a1
                results.data[i]['A'][t, 1] = a2
                results.data[i]['R'][t] = r
                results.data[i]['RPE'][t] = rpe
                results.data[i]['Q'][:, :, t] = Q


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

        if group_id is not None:
            results.data_mcmc['G'] = np.array([group_id]*nsubjects)

        return results
