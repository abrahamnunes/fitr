# -*- coding: utf-8 -*-
import numpy as np
from fitr.environments.graph import Graph

class MouthTask(Graph):
    """ The Pizzagalli reward sensitivity signal-detection task """
    def __init__(self,rng=np.random.RandomState()):
        T = np.zeros((2, 4, 4))
        #[u,x',x]
        T[0,2,0] = 0.3   # S=Lean, A=Lean, S'=Rewarded State
        T[0,3,0] = 0.7   # S=Lean, A=Lean, S'=Unrewarded state
        T[1,2,0] = 0.0   # S=Lean, A=Rich, S'=Rewarded state
        T[1,3,0] = 1.0   # S=Lean, A=Rich, S'=Rewarded state

        T[0,2,1] = 0.0   # S=Rich, A=Lean, S'=Rewarded state
        T[0,3,1] = 1.0   # S=Rich, A=Lean, S'=Unrewarded state
        T[1,2,1] = 0.75  # S=Rich, A=Rich, S'=Rewarded state
        T[1,3,1] = 0.25  # S=Rich, A=Rich, S'=Unrewarded state

        p_start = np.array([0.5,0.5,0.,0.])
        R = np.array([0.,0.,1.,0])
        xend = np.array([0.,0.,1.,1.])
        slabs = [r'$s_{0}$', r'$s_{1}$', r'Win', r'Nil']
        alabs = [r'$a_{0}$', r'$a_{1}$']
        taskname = 'Reward Signal Detection Task'
        super().__init__(T,R,xend,p_start,state_labels=slabs,
                         action_labels=alabs, label=taskname,
                         rng=rng)
        self.Ti[0,2,1] = 1.
        self.Ti[1,2,1] = 1.
