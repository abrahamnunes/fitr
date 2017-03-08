"""
Log-likelihood functions for each task.
"""

import numpy as np
from .utils import logsumexp

#-------------------------------------------------------------------------------
#
#   TWO-STEP TASK
#       Log-likelihood functions for various models of the two-step task
#
#-------------------------------------------------------------------------------

class twostep_ll(object):
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

            loglik = loglik + cr*Q[s2,a2] - logsumexp(cr*Q[s2,:])
            loglik = loglik + cr*Q[s1,a1] - logsumexp(cr*Q[s1,:])

            # Update model-free values
            Qmf[s2, a2] = Qmf[s2, a2] + lr*(r - Qmf[s2, a2])
            Qmf[s1, a1] = Qmf[s1, a1] + lr*(Qmf[s2, a2]-Qmf[s1, a1])

            # Update model based values
            Qmb[0, a1] = self.ptrans[1]*np.max(Qmf[1,:]) + self.ptrans[0]*np.max(Qmf[2,:])
            Qmb[0, a2] = self.ptrans[1]*np.max(Qmf[2,:]) + self.ptrans[0]*np.max(Qmf[1,:])

            # Linear combination of MF and MB
            Q = w*Qmb + (1-w)*Qmf


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

            loglik = loglik + cr*Q[s2,a2] - logsumexp(cr*Q[s2,:])
            loglik = loglik + cr*Q[s1,a1] - logsumexp(cr*Q[s1,:])

            # Update model-free values
            Qmf[s2, a2] = Qmf[s2, a2] + lr*(r - Qmf[s2, a2])
            Qmf[s1, a1] = Qmf[s1, a1] + lr*(Qmf[s2, a2]-Qmf[s1, a1])*et

            # Update model based values
            Qmb[0, a1] = self.ptrans[1]*np.max(Qmf[1,:]) + self.ptrans[0]*np.max(Qmf[2,:])
            Qmb[0, a2] = self.ptrans[1]*np.max(Qmf[2,:]) + self.ptrans[0]*np.max(Qmf[1,:])

            # Linear combination of MF and MB
            Q = w*Qmb + (1-w)*Qmf


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
