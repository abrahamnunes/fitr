# -*- coding: utf-8 -*-
import numpy as np
from fitr.environments.graph import Graph

class TwoArmedBandit(Graph):
    """ A simple 2-armed bandit task
    """
    def __init__(self, rng=np.random.RandomState()):
        T = np.zeros((2, 3, 3))
        T[0,1,0] = 0.8      # These end up being reward probabilities
        T[0,2,0] = 0.2      #  because states are deterministically rewarded
        T[1,1,0] = 0.2
        T[1,2,0] = 0.8

        p_start = np.array([1.,0.,0.])
        R = np.array([0.,0.,1.])
        xend = np.array([0.,1.,1.])
        super().__init__(T,R,xend,p_start,rng=rng)
