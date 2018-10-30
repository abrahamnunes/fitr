# -*- coding: utf-8 -*-
import numpy as np
from fitr.environments.graph import Graph

class OrthogonalGoNoGo(Graph):
    """ The Orthogonal GoNogo task from Guitart-Masip et al. (2012)
    """
    def __init__(self,rng=np.random.RandomState()):
        T = np.zeros((2,7,7))
        T[0,4,0] = 0.8  # S = Go to avoid loss, A=NoGo, S'= Loss
        T[0,5,0] = 0.2  # S = Go to avoid loss, A=NoGo, S'= Nothing
        T[1,4,0] = 0.2  # S = Go to avoid loss, A=Go,   S'= Loss
        T[1,5,0] = 0.8  # S = Go to avoid loss, A=Go,   S'= Nothing

        T[0,4,1] = 0.2  # S = NoGo to avoid loss, A=NoGo, S'= Loss
        T[0,5,1] = 0.8  # S = NoGo to avoid loss, A=NoGo, S'= Nothing
        T[1,4,1] = 0.8  # S = NoGo to avoid loss, A=Go,   S'= Loss
        T[1,5,1] = 0.2  # S = NoGo to avoid loss, A=Go,   S'= Nothing

        T[0,5,2] = 0.8  # S = Go to win, A=NoGo, S'= Nothing
        T[0,6,2] = 0.2  # S = Go to win, A=NoGo, S'= Win
        T[1,5,2] = 0.2  # S = Go to win, A=Go,   S'= Nothing
        T[1,6,2] = 0.8  # S = Go to win, A=Go,   S'= Win

        T[0,5,3] = 0.2  # S = NoGo to win, A=NoGo, S'= Nothing
        T[0,6,3] = 0.8  # S = NoGo to win, A=NoGo, S'= Win
        T[1,5,3] = 0.8  # S = NoGo to win, A=Go,   S'= Nothing
        T[1,6,3] = 0.2  # S = NoGo to win, A=Go,   S'= Win

        R            = np.array([0,0,0,0,-1,0,1])
        end_states   = np.array([0,0,0,0, 1,1,1])
        p_start      = np.array([1/4,1/4,1/4,1/4,0,0,0])
        taskname     = 'Orthogonal Go-NoGo Task'
        slabs        = [r'$s_{GA}$', r'$s_{NA}$', r'$s_{GW}$', r'$s_{NW}$',
                        'Loss', '0', 'Win']
        alabs        = [r'$a_{N}$', r'$a_{G}$']
        super().__init__(T,R,end_states,p_start,state_labels=slabs,
                         action_labels=alabs, label=taskname,
                         rng=rng)
