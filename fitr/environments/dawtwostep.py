# -*- coding: utf-8 -*-
import numpy as np
from fitr.environments.graph import Graph
from fitr.environments.utils import reward_reflection

class DawTwoStep(Graph):
    """ An implementation of the Two-Step Task from Daw et al. (2011).

    Arguments:

        mu: `float` identifying the drift of the reward-determining Gaussian random walks
        sd: `float` identifying the standard deviation of the reward-determining Gaussian random walks

    """
    def __init__(self, mu=0, sd=0.025,rng=np.random.RandomState()):
        T = np.zeros((2,5,5))
        T[0,1,0] = 0.7
        T[0,2,0] = 0.3
        T[1,1,0] = 0.3
        T[1,2,0] = 0.7

        T[0,3,1] = 0.2
        T[1,3,1] = 0.4
        T[0,3,2] = 0.6
        T[1,3,2] = 0.8

        T[0,4,1] = 1 - T[0,3,1]
        T[1,4,1] = 1 - T[1,3,1]
        T[0,4,2] = 1 - T[0,3,2]
        T[1,4,2] = 1 - T[1,3,2]

        p_start    = np.array([1,0,0,0,0])
        R          = np.array([0,0,0,1,0])
        end_states = np.array([0,0,0,1,1])
        slabs = [r'$s_A$', r'$s_B$', r'$s_C$', 'Win', 'Nil']
        alabs = [r'$a_0$', r'$a_1$']
        taskname = 'Two-Step Task'

        edge_labels = T.astype(np.str)
        edge_labels[0,3,1] = r'$P_{win}^{a_0^{s_B}}$'
        edge_labels[1,3,1] = r'$P_{win}^{a_1^{s_B}}$'
        edge_labels[0,3,2] = r'$P_{win}^{a_0^{s_C}}$'
        edge_labels[1,3,2] = r'$P_{win}^{a_1^{s_C}}$'
        edge_labels[0,4,1] = r'$1-P_{win}^{a_0^{s_B}}$'
        edge_labels[1,4,1] = r'$1-P_{win}^{a_1^{s_B}}$'
        edge_labels[0,4,2] = r'$1-P_{win}^{a_0^{s_C}}$'
        edge_labels[1,4,2] = r'$1-P_{win}^{a_1^{s_C}}$'
        super().__init__(T,R,end_states,p_start,f_reward=self.f_reward,
                         state_labels=slabs, action_labels=alabs,
                         label=taskname, edge_labels=edge_labels,
                         rng=rng)

        self.mu = mu
        self.sd = sd
        self.reward_paths = T[:,3,1:3].reshape(1, -1)

    def f_reward(self, R, x):
        rout = self.rng.binomial(1,np.einsum('s,s->',self.R,x))

        # Update the reward paths
        self.T[0,3,1] += self.rng.normal(self.mu, self.sd)
        self.T[1,3,1] += self.rng.normal(self.mu, self.sd)
        self.T[0,3,2] += self.rng.normal(self.mu, self.sd)
        self.T[1,3,2] += self.rng.normal(self.mu, self.sd)

        self.T[0,3,1] = reward_reflection(self.T[0,3,1], 0.2, 0.8)
        self.T[1,3,1] = reward_reflection(self.T[1,3,1], 0.2, 0.8)
        self.T[0,3,2] = reward_reflection(self.T[0,3,2], 0.2, 0.8)
        self.T[1,3,2] = reward_reflection(self.T[1,3,2], 0.2, 0.8)

        self.T[0,4,1] = 1 - self.T[0,3,1]
        self.T[1,4,1] = 1 - self.T[1,3,1]
        self.T[0,4,2] = 1 - self.T[0,3,2]
        self.T[1,4,2] = 1 - self.T[1,3,2]

        tstack = np.array([self.T[0,3,1], self.T[1,3,1], self.T[0,3,2], self.T[1,3,2]])
        self.reward_paths = np.vstack((self.reward_paths, tstack))
        return rout

    def plot_reward_paths(self, outfile=None, outfiletype='pdf', figsize=None):
        if figsize is None:
            figsize = (8,3)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel('Trial')
        ax.set_ylabel(r'p(Reward)')
        ax.plot(self.reward_paths)
        plt.tight_layout()

        if outfile is None:
            plt.show()
        else:
            if outfiletype == 'png':
                plt.savefig(outfile+'.png', dpi=350, bbox_inches='tight')
            else:
                plt.savefig(outfile+'.'+outfiletype, bbox_inches='tight')
