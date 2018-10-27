# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import multivariate_normal
from fitr.environments.graph import Graph
from fitr.environments.utils import reward_reflection

class KoolTwoStepWithAvoidance(Graph):
    """
    Modified from Kool, Cushman, & Gershman 2016.
    """
    def __init__(self, mu=0, sd=2, path_corr=0., reward_lb=-4, reward_ub=5,
                 rng=np.random.RandomState()):
        T = np.zeros([3,8,8])
        T[0,2,0] = 1.
        T[1,3,0] = 1.
        T[2,4,0] = 1.

        T[0,2,1] = 1.
        T[1,3,1] = 1.
        T[2,4,1] = 1.

        T[0,6,2] = 1.
        T[1,2,2] = 1.
        T[2,4,2] = 1.

        T[0,7,3] = 1.
        T[1,3,3] = 1.
        T[2,4,3] = 1.

        T[0,4,4] = 1.
        T[1,4,4] = 1.
        T[2,5,4] = 1.

        p_start    = np.array([0.5,0.5,0.,0.,0.,0.,0.])
        R          = np.array([0.,0.,0.,0.,0.,0.,0.3,0.7])
        end_states = np.array([0.,0.,0.,0.,0.,1.,1.,1.])
        slabs = [r'$s_A$', r'$s_B$', r'$s_C$', r'$s_D$',r'$s_E$', r'$s_F$', r'$s_G$', r'$s_H$']
        alabs = [r'$a_0$', r'$a_1$', r'$a_2$']
        super().__init__(T,R,end_states,p_start,f_reward=self.f_reward,
                         state_labels=slabs, action_labels=alabs,rng=rng)

        self.mu = mu
        self.sd = sd
        self.reward_lb = reward_lb
        self.reward_ub = reward_ub
        self.reward_hx = R
        self.path_corr = path_corr
        self.C = np.array([[sd**2, path_corr*(sd**2)], [path_corr*(sd**2), sd**2]])
        self.mvn = multivariate_normal(mean=np.zeros(2), cov=self.C)
        #self.mvn.random_seed = self.rng.get_state()[1][0]

    def f_reward(self, R, x):
        self.R[-2:] = self.R[-2:] + self.mvn.rvs()
        self.R[-2:] = reward_reflection(self.R[-2:], self.reward_lb, self.reward_ub)
        self.reward_hx = np.vstack((self.reward_hx, self.R))
        return np.einsum('s,s->',self.R,x)
