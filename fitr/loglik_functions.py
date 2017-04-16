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
Module containing log-likelihood functions for different models (as classes).
"""
import numpy as np
from .utils import logsumexp

# ------------------------------------------------------------------------------
#
#   TWO-STEP TASK
#       Log-likelihood functions for various models of the two-step task
#
# ------------------------------------------------------------------------------

class twostep_ll(object):
    """
    Likelihood functions for the two-step task [1]

    Attributes
    ----------
    ptrans : ndarray
        Transition probabilities for the two step task.

    Methods
    -------
    lr_cr_w(self, params, states, actions, rewards)
        Model with learning rate, choice randomness, and MB/MF weight
    lr_cr_w_p(self, params, states, actions, rewards)
        Model with learning rate, choice randomness, MB/MF weight, and perseveration parameter
    lr_cr_et_w_p(self, params, states, actions, rewards)
        Model with learning rate, choice randomness, eligibility trace, MB/MF weight, and perseveration parameter
    lr_cr_et_w(self, params, states, actions, rewards)
        Model with learning rate, choice randomness, and MB/MF weight
    lr2_cr3(self, params, states, actions, rewards)
        Model with learning rates for step 1 and 2, choice randomness for both MB and MF, as well as a choice randomness parameter for second step
    dummy(self, params, states, actions, rewards)
        Model with only a choice randomness, and no learning.

    References
    ----------
    [1] Daw, N.D. et al. (2011) Model-based influences on humans’ choices and striatal prediction errors. Neuron 69, 1204–1215
    """
    def __init__(self, ptrans=0.7, rewards=[1, 0]):
        self.ptrans = np.array([1-ptrans, ptrans])

    def lr_cr_w(self, params, states, actions, rewards):
        """
        Likelihood function for model containing parameters (A) learning rate, (B) choice randomness, and (C) MB weight parameter
        """
        # Initialize parameters
        lr = params[0]
        cr = params[1]
        w  = params[2]

        # Initialize log-likelihood
        loglik = 0

        # Initialize Q arrays
        Q   = np.zeros([3, 2])
        Qmb = np.zeros([3, 2])
        Qmf = np.zeros([3, 2])

        ntrials = np.shape(states)[0]
        for t in range(ntrials):
            s1 = int(states[t,0])
            s2 = int(states[t,1])
            a1 = int(actions[t,0])
            a2 = int(actions[t,1])
            r = rewards[t]

            loglik = loglik + cr*Q[s1,a1] - logsumexp(cr*Q[s1,:])
            loglik = loglik + cr*Q[s2,a2] - logsumexp(cr*Q[s2,:])

            # Update model-free values
            Qmf[s1, a1] = Qmf[s1, a1] + lr*(Qmf[s2, a2]-Qmf[s1, a1])

            rpe = r - Qmf[s2, a2]
            Qmf[s2, a2] = Qmf[s2, a2] + lr*rpe
            Qmf[s1, a1] = Qmf[s1, a1] + lr*rpe

            # Update model based values
            Qmb[0, 0] = 0.7*np.max(Qmf[1,:]) + 0.3*np.max(Qmf[2,:])
            Qmb[0, 1] = 0.7*np.max(Qmf[2,:]) + 0.3*np.max(Qmf[1,:])

            # Linear combination of MF and MB
            Q = w*Qmb + (1-w)*Qmf


        return loglik

    def lr_cr_w_p(self, params, states, actions, rewards):
        """
        Likelihood function for model containing parameters (A) learning rate, (B) choice randomness, (C) MB weight parameter, and (D) perseveration
        """
        # Initialize parameters
        lr = params[0]
        cr = params[1]
        w = params[2]
        p = params[3]

        # Initialize log-likelihood
        loglik = 0

        # Initialize Q arrays
        Q   = np.zeros([3, 2])
        Qmb = np.zeros([3, 2])
        Qmf = np.zeros([3, 2])

        # Initialize a variable representing the last action taken
        #   This is set to 10 in order for the initial perseveration value
        #   to equal 0
        a0 = 10

        ntrials = np.shape(states)[0]
        for t in range(ntrials):
            s1 = int(states[t,0])
            s2 = int(states[t,1])
            a1 = int(actions[t,0])
            a2 = int(actions[t,1])
            r = rewards[t]

            loglik = loglik + cr*Q[s1,a1] - logsumexp(cr*Q[s1,:])
            loglik = loglik + cr*Q[s2,a2] - logsumexp(cr*Q[s2,:])

            # Update model-free values
            Qmf[s1, a1] = Qmf[s1, a1] + lr*(Qmf[s2, a2]-Qmf[s1, a1])

            rpe = r - Qmf[s2, a2]
            Qmf[s2, a2] = Qmf[s2, a2] + lr*rpe
            Qmf[s1, a1] = Qmf[s1, a1] + lr*rpe

            # Update model based values
            Qmb[0, 0] = 0.7*np.max(Qmf[1,:]) + 0.3*np.max(Qmf[2,:])
            Qmb[0, 1] = 0.7*np.max(Qmf[2,:]) + 0.3*np.max(Qmf[1,:])

            # Perseveration
            persev = p * (a1 == a0)

            # Linear combination of MF and MB
            Q = w*Qmb + (1-w)*Qmf + persev

            # Set a0 to the most recent action
            a0 = a1


        return loglik

    def lr_cr_et_w_p(self, params, states, actions, rewards):
        """
        Likelihood function for model containing parameters (A) learning rate, (B) choice randomness, (C) eligibility trace parameter, (D) MB weight parameter, and (E) perseveration
        """
        # Initialize parameters
        lr = params[0]
        cr = params[1]
        et = params[2]
        w = params[3]
        p = params[4]

        # Initialize log-likelihood
        loglik = 0

        # Initialize Q arrays
        Q   = np.zeros([3, 2])
        Qmb = np.zeros([3, 2])
        Qmf = np.zeros([3, 2])

        # Initialize a variable representing the last action taken
        #   This is set to 10 in order for the initial perseveration value
        #   to equal 0
        a0 = 10

        ntrials = np.shape(states)[0]
        for t in range(ntrials):
            s1 = int(states[t,0])
            s2 = int(states[t,1])
            a1 = int(actions[t,0])
            a2 = int(actions[t,1])
            r = rewards[t]

            loglik = loglik + cr*Q[s1,a1] - logsumexp(cr*Q[s1,:])
            loglik = loglik + cr*Q[s2,a2] - logsumexp(cr*Q[s2,:])

            # Update model-free values
            Qmf[s1, a1] = Qmf[s1, a1] + lr*(Qmf[s2, a2]-Qmf[s1, a1])

            rpe = r - Qmf[s2, a2]
            Qmf[s2, a2] = Qmf[s2, a2] + lr*rpe
            Qmf[s1, a1] = Qmf[s1, a1] + lr*rpe*et

            # Update model based values
            Qmb[0, 0] = 0.7*np.max(Qmf[1,:]) + 0.3*np.max(Qmf[2,:])
            Qmb[0, 1] = 0.7*np.max(Qmf[2,:]) + 0.3*np.max(Qmf[1,:])

            # Perseveration
            persev = p * (a1 == a0)

            # Linear combination of MF and MB
            Q = w*Qmb + (1-w)*Qmf + persev

            # Set a0 to the most recent action
            a0 = a1


        return loglik

    def lr_cr_et_w(self, params, states, actions, rewards):
        """
        Likelihood function for model containing parameters (A) learning rate, (B) choice randomness, (C) eligibility trace, and (D) MB weight parameter
        """
        # Initialize parameters
        lr = params[0]
        cr = params[1]
        et = params[2]
        w  = params[3]

        # Initialize log-likelihood
        loglik = 0

        # Initialize Q arrays
        Q   = np.zeros([3, 2])
        Qmb = np.zeros([3, 2])
        Qmf = np.zeros([3, 2])

        ntrials = np.shape(states)[0]
        for t in range(ntrials):
            s1 = int(states[t,0])
            s2 = int(states[t,1])
            a1 = int(actions[t,0])
            a2 = int(actions[t,1])
            r = rewards[t]

            loglik = loglik + cr*Q[s1,a1] - logsumexp(cr*Q[s1,:])
            loglik = loglik + cr*Q[s2,a2] - logsumexp(cr*Q[s2,:])

            # Update model-free values
            Qmf[s1, a1] = Qmf[s1, a1] + lr*(Qmf[s2, a2]-Qmf[s1, a1])

            rpe = r - Qmf[s2, a2]
            Qmf[s2, a2] = Qmf[s2, a2] + lr*rpe
            Qmf[s1, a1] = Qmf[s1, a1] + lr*rpe*et

            # Update model based values
            Qmb[0, 0] = 0.7*np.max(Qmf[1,:]) + 0.3*np.max(Qmf[2,:])
            Qmb[0, 1] = 0.7*np.max(Qmf[2,:]) + 0.3*np.max(Qmf[1,:])

            # Linear combination of MF and MB
            Q = w*Qmb + (1-w)*Qmf


        return loglik

    def lr2_cr3(self, params, states, actions, rewards):
        """
        Likelihood function for model containing parameters (A) step 1 learning rate, (B) step 2 learning rate, (C) MB choice randomness, (D) MF choice randomness, (E) step 2 choice randomness
        """
        # Initialize parameters
        lr_1 = params[0]
        lr_2 = params[1]
        cr_mb = params[2]
        cr_mf  = params[3]
        cr_2 = params[4]

        # Initialize log-likelihood
        loglik = 0

        # Initialize Q arrays
        Qmb = np.zeros([3, 2])
        Qmf = np.zeros([3, 2])

        ntrials = np.shape(states)[0]
        for t in range(ntrials):
            s1 = int(states[t,0])
            s2 = int(states[t,1])
            a1 = int(actions[t,0])
            a2 = int(actions[t,1])
            r = rewards[t]

            Q1 = cr_mb*Qmb[s1,a1] + cr_mf*Qmf[s1, a1]
            loglik = loglik + cr_2*Qmf[s2,a2] - logsumexp(cr_2*Qmf[s2,:])
            loglik = loglik + Q1 - logsumexp(Q1)

            # Update model-free values
            Qmf[s2, a2] = Qmf[s2, a2] + lr_2*(r - Qmf[s2, a2])
            Qmf[s1, a1] = Qmf[s1, a1] + lr_1*(Qmf[s2, a2]-Qmf[s1, a1])

            # Update model based values
            Qmb[0, a1] = self.ptrans[1]*np.max(Qmf[1,:]) + self.ptrans[0]*np.max(Qmf[2,:])
            Qmb[0, a2] = self.ptrans[1]*np.max(Qmf[2,:]) + self.ptrans[0]*np.max(Qmf[1,:])


        return loglik

    def dummy(self, params, states, actions, rewards):
        """
        Likelihood function without learning
        """

        # Initialize parameters
        cr = params[0]

        # Initialize log-likelihood
        loglik = 0

        # Initialize Q arrays
        Q   = np.zeros([3, 2])
        Qmb = np.zeros([3, 2])
        Qmf = np.zeros([3, 2])

        ntrials = np.shape(states)[0]
        for t in range(ntrials):
            s1 = int(states[t,0])
            s2 = int(states[t,1])
            a1 = int(actions[t,0])
            a2 = int(actions[t,1])

            loglik = loglik + cr*Q[s2,a2] - logsumexp(cr*Q[s2,:])
            loglik = loglik + cr*Q[s1,a1] - logsumexp(cr*Q[s1,:])

        return loglik
#-------------------------------------------------------------------------------
#
#   BANDIT
#       Log-likelihood functions for various models of a bandit task
#
#-------------------------------------------------------------------------------

class bandit_ll(object):
    """
    Log-likelihood function for the bandit task.

    Attributes
    ----------
    narms : int > 0
        Number of arms in the bandit task

    Methods
    -------
    lr_cr(self, params, states, actions, rewards)
        Model with learning rate, and choice randomness
    lr_cr_rs(self, params, states, actions, rewards)
        Model with learning rate, choice randomness, and reward sensitivity
    lrp_lrn_cr(self, params, states, actions, rewards)
        Model with learning rate for positive RPE, learning rate for negative RPE, and a choice randomness parameter
    lrp_lrn_cr_rs(self, params, states, actions, rewards)
        Model with learning rate for positive RPE, learning rate for negative RPE, a choice randomness parameter, and reward sensitivity parameter
    dummy(self, params, states, actions, rewards)
        Model with no learning
    """
    def __init__(self, narms=2):
        self.narms=narms

    def lr_cr(self, params, states, actions, rewards):
        """
        Likelihood function containing parameters (A) learning rate, and (B) choice randomness
        """
        lr = params[0]
        cr = params[1]

        ntrials = len(actions)
        Q = np.zeros(self.narms)
        loglik = 0

        for t in range(0, ntrials):
            a = int(actions[t])
            r = rewards[t]
            loglik = loglik + cr*Q[a] - logsumexp(cr*Q)
            Q[a] = Q[a] + lr*(r - Q[a])

        return loglik

    def lr_cr_rs(self, params, states, actions, rewards):
        """
        Likelihood function containing parameters (A) learning rate, (B) choice randomness, and (C) reward sensitivity
        """
        lr = params[0]
        cr = params[1]
        rs = params[2]

        ntrials = len(actions)
        Q = np.zeros(self.narms)
        loglik = 0

        for t in range(0, ntrials):
            a = int(actions[t])
            r = rewards[t]
            loglik = loglik + cr*Q[a] - logsumexp(cr*Q)
            Q[a] = Q[a] + lr*(rs*r - Q[a])

        return loglik

    def lrp_lrn_cr(self, params, states, actions, rewards):
        """
        Likelihood function containing parameters (A) positive learning rate, (B) negative learning rate, and (C) choice randomness
        """
        lrp = params[0]
        lrn = params[1]
        cr  = params[2]

        ntrials = len(actions)
        Q = np.zeros(self.narms)
        loglik = 0

        for t in range(0, ntrials):
            a = int(actions[t])
            r = rewards[t]
            loglik = loglik + cr*Q[a] - logsumexp(cr*Q)

            rpe = r-Q[a]
            if rpe >=0:
                lr = lrp
            else:
                lr = lrn

            Q[a] = Q[a] + lr*rpe

        return loglik

    def lrp_lrn_cr_rs(params, states, actions, rewards):
        """
        Likelihood function containing parameters (A) positive learning rate, (B) negative learning rate, (c) choice randomness, and (D) reward sensitivity
        """
        lrp = params[0]
        lrn = params[1]
        cr  = params[2]
        rs  = params[3]

        ntrials = len(actions)
        Q = np.zeros(self.narms)
        loglik = 0

        for t in range(0, ntrials):
            a = int(actions[t])
            r = rewards[t]
            loglik = loglik + cr*Q[a] - logsumexp(cr*Q)

            rpe = rs*r-Q[a]
            if rpe >=0:
                lr = lrp
            else:
                lr = lrn

            Q[a] = Q[a] + lr*rpe

        return loglik

    def dummy(self, params, states, actions, rewards):
        """
        Likelihood function without any learning
        """
        cr  = params[0]

        ntrials = len(actions)
        Q = np.zeros(self.narms)
        loglik = 0

        for t in range(0, ntrials):
            a = int(actions[t])
            loglik = loglik + cr*Q[a] - logsumexp(cr*Q)

        return loglik
