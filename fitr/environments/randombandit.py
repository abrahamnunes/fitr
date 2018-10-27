# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import circulant
from fitr.environments.graph import Graph

#===============================================================================
#   FUNCTIONS FOR AUTOMATIC TASK GENERATOR
#===============================================================================

def initialize_context_action_dependencies(nactions, nstates, noutcomes):
    return np.hstack((np.ones((nactions, nstates)), np.zeros((nactions, noutcomes))))

def prune_context_actions(Z, min_actions_per_context):
    nactions = Z.shape[0]
    prune_mtx = np.zeros(Z.shape)
    if min_actions_per_context < nactions:
        done = False
        while not done:
            for j in range(Z.shape[1]):
                ntoprune = np.random.randint(0, (nactions-min_actions_per_context)+1)
                pvals = np.ones(nactions)/nactions
                prune_idx = np.random.multinomial(ntoprune, pvals=pvals)
                prune_mtx[:, j] = np.greater(prune_idx, 0)
            if not np.any(np.equal(prune_mtx.sum(1), 0)): done = True
        return np.greater(Z - prune_mtx, 0.).astype(np.int)
    elif min_actions_per_context > nactions:
        print ('Invalid number of minimum actions')
    else:
        return Z

def initialize_stochastic_matrix(A):
    """ Returns an action by state transition matrix """
    nactions, nstates = A.shape
    noutcomes = np.equal(A.sum(0), 0).sum()
    ncontexts = nstates - np.equal(A.sum(0), 0).sum()
    T = np.zeros((nactions, nstates, nstates))
    p = np.tile(np.expand_dims(A, 1), [1, noutcomes, 1])#/noutcomes
    T[:,-noutcomes:,:] = p
    return T

def append_outcomes(A, noutcomes):
    return np.hstack((A, np.zeros((A.shape[0], noutcomes))))

def make_controllable(T, alpha=0.01, shift_flip='shift'):
    """ Takes a transition graph and turns the `action->next state` transitions into probabilities

    Arguments:

        T : Transition tensor of type `ndarray((nactions, nstates, nstates))`
        alpha : The sharpness with which actions pick outcomes (`float > 0`)
        shift_flip : How to make differences in action-outcome contingencies between contexts. Options include `shift` and `flip`. Shifting takes the transition matrix from the prior action and shifts it, whereas the `flip` option does as its name suggests.

    """
    # TODO: Explain the `shift_flip` parameter further

    n = np.max(T.shape)
    m = T.shape[2]
    x = np.array([alpha**(i/(n-1)) for i in range(n)])
    x = circulant(x)
    x = x@x.T
    x = np.flip(x, axis=0)
    x = np.tile(x, [m, m])
    for k in range(T.shape[2]):
        if shift_flip=='shift':
            y = x[k:T.shape[0]+k, -T.shape[1]:]
        elif shift_flip == 'flip':
            y = x[:T.shape[0], -T.shape[1]:]
            if k % 2== 0:
                y = np.flip(y, axis=0)
        elif shift_flip == 'shiftandflip':
            y = x[k:T.shape[0]+k, -T.shape[1]:]
            if k % 2== 0:
                y = np.flip(y, axis=0)
        else:
            y = x[k:T.shape[0]+k, -(T.shape[1]+k):x.shape[1]-k]
        T[:,:,k] = y*T[:,:,k]
        Tsum = np.tile(np.sum(T[:,:,k], axis=1).reshape(-1, 1), [1, T[:,:,k].shape[1]])
        T[:,:,k] = np.ma.divide(T[:,:,k], Tsum, where=Tsum!=0)
    return T

def make_bandit_graph(nactions, noutcomes, nstates, min_actions_per_context, alpha, shift_flip):
    """ Creates a random transition tensor.

    Arguments:

        nactions: Integer number of actions in the task
        noutcomes: Integer number of total possible outcomes in the task
        nstates: Integer number of states (excluding outcomes) in the task
        min_actions_per_context: Different contexts may have more or fewer actions than others (never more than `nactions`). This variable describes the minimum number of actions allowed in a context.
        alpha: Sharpness of `action->outcome` contingencies
        shift_flip : How to make differences in action-outcome contingencies between contexts. Options include `shift` and `flip`. Shifting takes the transition matrix from the prior action and shifts it, whereas the `flip` option does as its name suggests.

    Returns:

        Transition tensor of type `ndarray((nactions, noutcomes, nstates))`
    """
    C = initialize_context_action_dependencies(nactions, nstates, noutcomes)
    C = prune_context_actions(C, min_actions_per_context)
    T = initialize_stochastic_matrix(C)
    T = make_controllable(T, alpha, shift_flip)
    return T


#===============================================================================
#   THE RANDOM BANDIT TASK
#===============================================================================

class RandomContextualBandit(Graph):
    """ Generates a random bandit task

    Arguments:

        nactions: Number of actions
        noutcomes: Number of outcomes
        nstates: Number of contexts
        min_actions_per_context: Different contexts may have more or fewer actions than others (never more than `nactions`). This variable describes the minimum number of actions allowed in a context.
        alpha:
        alpha_start:
        shift_flip:
        reward_lb: Lower bound for drifting rewards
        reward_ub: Upper bound for drifting rewards
        reward_drift: Values (`on` or `off`) determining whether rewards are allowed to drift
        drift_mu: Mean of the Gaussian random walk determining reward
        drift_sd: Standard deviation of Gaussian random walk determining reward
    """
    def __init__(self,
                 nactions,
                 noutcomes,
                 nstates,
                 min_actions_per_context=None,
                 alpha=0.1,
                 alpha_start=1.,
                 shift_flip='flip',
                 reward_lb=-1,
                 reward_ub=1,
                 reward_drift='off',
                 drift_mu=0.,
                 drift_sd=2.,
                 rng=np.random.RandomState()):
        self.nactions = nactions
        self.noutcomes = noutcomes
        self.nstates = nstates
        self.alpha = alpha
        self.alpha_start = alpha_start
        self.shift_flip = shift_flip
        self.min_actions_per_context = min_actions_per_context
        if self.min_actions_per_context is None:
            self.min_actions_per_context = self.nactions

        T = make_bandit_graph(nactions=self.nactions,
                              noutcomes=self.noutcomes,
                              nstates=self.nstates,
                              min_actions_per_context=self.min_actions_per_context,
                              alpha=self.alpha,
                              shift_flip=self.shift_flip)



        p_start = np.zeros(self.nstates + self.noutcomes)
        xend = np.zeros(self.nstates + self.noutcomes)
        R = np.zeros(self.nstates + self.noutcomes)

        p_start[:self.nstates]=np.array([alpha_start**(i/(self.nstates-1)) for i in range(self.nstates)])
        p_start = p_start/np.sum(p_start)
        xend[-self.noutcomes:] = 1.
        R[-self.noutcomes:] = np.linspace(reward_lb, reward_ub, self.noutcomes)

        # Reward drift properties
        self.mu = drift_mu
        self.sd = drift_sd
        self.reward_lb = reward_lb
        self.reward_ub = reward_ub
        self.reward_hx = R
        self.reward_drift = reward_drift
        if self.reward_drift == 'on':
            self.mu = drift_mu
            self.sd = drift_sd
        elif self.reward_drift == 'off':
            self.mu = np.zeros(self.noutcomes)
            self.sd = 1.
        self.C = np.eye(self.noutcomes)*self.sd
        self.mvn = multivariate_normal(mean=self.mu, cov=self.C)
        super().__init__(T,R,xend,p_start,f_reward=self.f_reward,rng=rng)

    def f_reward(self, R, x):
        if self.reward_drift == 'on':
            perturbation = self.mvn.rvs()
        elif self.reward_drift == 'off':
            perturbation = np.zeros(self.noutcomes)
        self.R[-self.noutcomes:] = self.R[-self.noutcomes:] + perturbation
        self.R[-self.noutcomes:] = reward_reflection(self.R[-self.noutcomes:], self.reward_lb, self.reward_ub)
        self.reward_hx = np.vstack((self.reward_hx, self.R))
        return np.einsum('s,s->',self.R,x)
