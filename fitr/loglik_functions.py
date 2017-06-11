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

References
----------
.. [Daw2011] Daw, N.D. et al. (2011) Model-based influences on humans’ choices and striatal prediction errors. Neuron 69, 1204–1215

Module Documentation
--------------------
"""
import warnings

import numpy as np
from .utils import logsumexp

# Warn user of the deprecated module
warnings.warn(
    "The loglik_functions module is deprecated, and will be removed in future releases. Please use similar capacities task-specific modules",
    DeprecationWarning
)

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
