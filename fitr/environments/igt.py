# -*- coding: utf-8 -*-
import numpy as np
from fitr.environments.graph import Graph

class IGT(Graph):
    """ Iowa Gambling Task """
    def __init__(self,rng=np.random.RandomState()):
        T = np.zeros((4, 5, 5))
        T[0,1,0] = 0.5
        T[0,3,0] = 0.5
        T[1,1,0] = 0.5
        T[1,3,0] = 0.5
        T[2,2,0] = 0.5
        T[2,4,0] = 0.5
        T[3,2,0] = 0.5
        T[3,4,0] = 0.5

        p_start = np.array([1., 0., 0., 0., 0.])
        R = np.array([0., 100., 50., -250., -50.])
        xend = np.array([0., 1., 1., 1., 1.])
        slabs = [r'$s_0$', '+$100', '+$50', '-$250', '-$50']
        alabs = [r'$a_A$', r'$a_B$', r'$a_C$', r'$a_D$']
        taskname = 'Iowa Gambling Task'
        super().__init__(T, R, xend, p_start, state_labels=slabs,
                         action_labels=alabs, label=taskname,
                         rng=rng)
