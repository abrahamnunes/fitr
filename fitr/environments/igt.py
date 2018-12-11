# -*- coding: utf-8 -*-
import numpy as np
from fitr.environments.graph import Graph

class IGT(Graph):
    """ Iowa Gambling Task """
    def __init__(self,rng=np.random.RandomState()):
        T = np.zeros((4, 7, 7))
        T[0,1,0] = 0.9 # Deck A Reward 100 (9/10)
        T[0,3,0] = 0.1 # Deck A Reward 100 + Loss -1250 = Net -1150 (1/10)
       
        T[1,1,0] = 0.5 # Deck B Reward 100 (1/2) 
        T[1,4,0] = 0.5 # Deck B Reward 100 + Loss -250 = net -150 (1/2)
        
        T[2,2,0] = 0.5 # Deck C Reward 50 (1/2)
        T[2,5,0] = 0.5 # Deck C Reward 50 + Loss 50 (1/2)
        
        T[3,2,0] = 0.9 # Deck D Reward 50 (1/2)
        T[3,6,0] = 0.1 # Deck D Reward 50 + Loss -250 = net -200 (1/10)

        p_start = np.array([1., 0., 0., 0., 0., 0., 0.])
        R = np.array([0., 100., 50., -1150., -150., 0., -200.])
        xend = np.array([0., 1., 1., 1., 1., 1., 1.])
        slabs = [r'$s_0$', '+$100', '+$50', '-$1,150', '-$150', '-$0', '-$200']
        alabs = [r'$a_A$', r'$a_B$', r'$a_C$', r'$a_D$']
        taskname = 'Iowa Gambling Task'
        super().__init__(T, R, xend, p_start, state_labels=slabs,
                         action_labels=alabs, label=taskname,
                         rng=rng)
