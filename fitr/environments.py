# -*- coding: utf-8 -*-

import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import circulant
from fitr.data import BehaviouralData
from fitr.data import merge_behavioural_data

class Graph(object):
    """ Base object that defines a reinforcement learning task.

    #### Definitions

    - $\mathbf x \in \mathcal X$ be a one-hot state vector, where $|\mathcal X|=n_x$
    - $\mathbf u \in \mathcal U$ be a one-hot action vector, where $|\mathcal U|=n_u$
    - $\mathsf T = p(\mathbf x_{t+1}|\mathbf x_t, \mathbf u_t)$ be a transition tensor
    - $p(\mathbf x)$ be a distribution over starting states
    - $\mathcal J: \mathcal X \\to \mathcal R$, where $\mathcal R \subseteq \mathbb R$ be a reward function

    Arguments:

        T: Transition tensor
        R: Vector of rewards for each state such that scalar reward $r_t = \mathbf r^\top \mathbf x$
        end_states: A vector $\{0, 1\}^{n_x}$ identifying which states terminate a trial (aka episode)
        p_start: Initial state distribution
        label: A string identifying a name for the task
        state_labels: A list or array of strings labeling the different states (for plotting purposes)
        action_labels: A list or array of strings labeling the different actions (for plotting purposes)
        rng: `np.random.RandomState` object
        f_reward: A function whose first argument is a vector of rewards for each state, and whose second argument is a state vector, and whose output is a scalar reward
        cmap: Matplotlib colormap for plotting.

    #### Notes

    There are two critical methods for the `Graph` class: `observation()` and `step`. All instances of a `Graph` must be able to call these functions. Let's say you have some bandit task `MyBanditTask` that inherits from `Graph`. To run such a task would look something like this:

    ``` python
    env = MyBanditTask()            # Instantiate your environment object
    agent = MyAgent()               # Some agent object (arbitrary, really)
    for t in range(ntrials): 
            x = env.observation()       # Samples initial state
            u = agent.action(x)         # Choose some action
            x_, r, done = agent.step(u) # Transition based on action
    ```

    What differentiates tasks are the transition tensor $\mathsf T$, starting state distribution $p(\mathbf x)$ and reward function $\mathcal J$ (which here would include the reward vector $\mathbf r$).
    """
    def __init__(self,
                 T=None,
                 R=None,
                 end_states=None,
                 p_start=None,
                 label=None,
                 state_labels=None,
                 action_labels=None,
                 edge_labels=None,
                 rng=np.random.RandomState(),
                 f_reward=lambda r, x: np.einsum('s,s->',r,x),
                 cmap=plt.get_cmap('Greys_r')):
        self.T             = T             # Transition matrix (AxSxS)
        self.Ti            = T             # ID of edges for graph plot
        self.Tlab          = edge_labels   # Labels for edges
        self.R             = R             # Reward vector
        self.end_states    = end_states    # Indicates terminal states
        self.label         = label         # Label for the task
        self.state_labels  = state_labels  # Labels for states
        self.action_labels = action_labels # Labels for actions
        self.nactions      = T.shape[0]    # Number of actions
        self.nstates       = T.shape[1]    # Number of states
        self.p_start       = p_start       # Probabilities over start states
        self.f_reward      = f_reward      # Reward generating function
        self.t             = 0             # Trial counter
        self.rng           = rng           # Random number generator
        self.cmap          = cmap          # Colormap for plots
        self.done          = True          # Prevents 'step' unless initialized

        # Count number of outcomes
        self.noutcomes = np.sum(self.end_states).astype(np.int)

        # Threshold to identify the edges in the graph
        self.Ti = np.greater(self.Ti, 0.).astype(np.float32)

        # Set edge labels to transition matrix if none supplied
        if self.Tlab is None: self.Tlab = self.T.round(3).astype(np.str)

        if self.state_labels is None: self.make_state_labels()
        if self.action_labels is None: self.make_action_labels()

        # Graph ops
        self.make_digraph()
        self.make_undirected_graph()
        self.laplacian_matrix_decomposition()
        self.adjacency_matrix_decomposition()

    def observation(self):
        """ Samples an initial state from the start-state distribution $p(\mathbf x)$

        $$
        \mathbf x_0 \sim p(\mathbf x)
        $$

        Returns:

            A one-hot vector `ndarray((nstates,))` indicating the starting state.

        Examples:

            ```python
            x = env.observation()
            ```
        """
        self.done = False
        self.state = self.rng.multinomial(1, self.p_start)
        return self.state

    def random_action(self):
        """ Samples a random one-hot action vector uniformly over the action space.

        Useful for testing that your environment works, without having to create an agent.

        $$
        \mathbf u \sim \mathrm{Multinomial}\Big(1, \mathbf p=\{p_i = \\frac{1}{|\mathcal U|}\}_{i=1}^{|\mathcal U|}\Big)
        $$

        Returns:

            A one-hot action vector of type `ndarray((nactions,))`

        Examples:

            ```python
            u = env.random_action()
            ```

        """
        return self.rng.multinomial(1, [1/self.nactions]*self.nactions)

    def step(self, action):
        """ Executes a state transition in the environment.

        Arguments:

            action : A one-hot vector of type `ndarray((naction,))` indicating the action selected at the current state.

        Returns:

            A 3-tuple representing the next state (`ndarray((noutcomes,))`), scalar reward, and whether the current step terminates a trial (`bool`).

        Raises:

            `RuntimeError` if `env.observation()` not called after a previous `env.step(...)` call yielded a terminal state.
        """
        if not self.done:
            pstate = np.einsum('axs,s,a->x', self.T, self.state, action)
            self.state = self.rng.multinomial(1, pstate)
            r = self.f_reward(self.R, self.state)
            self.t += 1
            if np.einsum('s,s->',self.state,self.end_states)>0.: self.done=True
            return self.state, r, self.done
        else:
            raise RuntimeError('Must re-initialize with `observe` at each episode')

    def make_state_labels(self):
        """ Creates labels for the states (for plotting) if none provided """
        self.state_labels = []
        for i in range(self.nstates):
            self.state_labels.append(r'$s_%s$' %i)

    def make_action_labels(self):
        """ Creates labels for the actions (for plotting) if none provided """
        self.action_labels = []
        for i in range(self.nactions):
            self.action_labels.append(r'$a_%s$' %i)

    def make_digraph(self):
        """ Creates a `networkx` `DiGraph` object from the transition tensor for the purpose of plotting and some other analyses. """
        self.start_statelist = np.argwhere(np.equal(np.sum(np.sum(self.Ti, 0), 1), 0)).flatten()
        self.end_statelist   = np.argwhere(np.equal(np.sum(np.sum(self.Ti, 0), 0), 0)).flatten()

        self.G = nx.DiGraph()
        self.G.add_node('START')
        self.G.add_node('END')
        self.graphparams = {}
        self.graphparams['valmap'] = {'START': 0, 'END': 0}
        self.graphparams['edgelabels'] = {}
        for s in range(self.nstates):
            snode = self.state_labels[s] #r'$s_%s$' %s
            self.graphparams['valmap'][snode] = 1
            self.G.add_node(snode)
            if s in self.start_statelist:
                self.G.add_edge('START', snode)
                self.graphparams['edgelabels'][('START', snode)] = '%s' %self.p_start[s] #r'$P_{s_%s}=%s$' %(s,pstart[s])
            if s in self.end_statelist:
                self.G.add_edge(snode, 'END')
                self.graphparams['edgelabels'][(snode, 'END')] = '1' #r'$P_{End}=1$'
            for a in range(int(np.sum(np.greater(np.sum(self.Ti[:,:,s], axis=1), 0)))):
                snode_sup = r'$^{(%s)}$' %re.sub('\$', '', snode) #r'$a_{%s}^{(s_%s)}$' %(a,s)
                anode = self.action_labels[a] + snode_sup
                self.graphparams['valmap'][anode] = 2
                self.G.add_node(anode)
                self.G.add_edge(snode, anode)
                self.graphparams['edgelabels'][(snode, anode)] = r'$\epsilon$'
                next_states = np.argwhere(np.greater(self.Ti[a,:,s], 0)).flatten()
                for s_ in next_states:
                    s_node = self.state_labels[s_] #r'$s_%s$' %s_
                    self.graphparams['valmap'][s_node] = 1
                    if s_node not in self.G.nodes:
                        self.G.add_node(s_node)
                    self.G.add_edge(anode, s_node)
                    self.graphparams['edgelabels'][(anode, s_node)] = '%s' %self.Tlab[a,s_,s]  #r'$P_{s_%s}^{s_%s a_{%s}^{(s_%s)}}=%s$' %(s_, s, a, s, T[a,s_,s])

        # Get some graph properties
        self.incidence_matrix = nx.incidence_matrix(self.G)

    def make_undirected_graph(self):
        """ Converts the DiGraph to undirected and computes some stats """
        self.G_u = nx.to_undirected(self.G)
        self.adjacency_matrix = nx.adjacency_matrix(self.G_u).todense()
        self.laplacian = nx.laplacian_matrix(self.G_u).todense()

    def laplacian_matrix_decomposition(self):
        """ Singular value decomposition of the graph Laplacian """
        u,s,v = np.linalg.svd(self.laplacian)
        self.laplacian_spectrum = s
        self.laplacian_eigenvectors = np.array(v)
        self.algebraic_connectivity = self.laplacian_spectrum[-2]

    def adjacency_matrix_decomposition(self):
        """ Singular value decomposition of the graph adjacency matrix """
        u,s,v = np.linalg.svd(self.adjacency_matrix)
        self.adjacency_spectrum = s
        self.adjacency_eigenvectors = np.array(v)

    def get_graph_depth(self):
        """ Returns the depth of the task graph.

        Calculated as the depth from `START` (pre-initial state) to `END` (which absorbs trial from all terminal states), minus 2 to account for the `START->node` & `node->END` transitions.

        Returns:

            An `int` identifying the depth of the current graph for a single trial of the task
        """
        return nx.shortest_path_length(self.G, source='START', target='END')-2

    def plot_graph(self,
                   figsize=None,
                   node_size=2000,
                   arrowsize=20,
                   lw=1.5,
                   font_size=12,
                   title=False,
                   outfile=None,
                   outfiletype='pdf'):
        """ Plots the directed graph of the task """
        if figsize is None:
            figsize = (20, 20)


        shape_map = {0:'s', 1: 's', 2: 'o'}
        edge_cmap = {0:'white', 1: 'k', 2: 'k'}
        color_map = {0:'white', 1: 'white', 2: 'white'}
        fontc_map = {0:'k', 1: 'k', 2: 'k'}
        nodecolors = [color_map[self.graphparams['valmap'][node]] for i,node in enumerate(self.G.nodes)] #color(valmap[node])
        edgecolors = [edge_cmap[self.graphparams['valmap'][node]] for i,node in enumerate(self.G.nodes)]
        fontcolors = [fontc_map[self.graphparams['valmap'][node]] for i,node in enumerate(self.G.nodes)]
        nodeshapes = [shape_map[self.graphparams['valmap'][node]] for i, node in enumerate(self.G.nodes)]

        fig, ax = plt.subplots(figsize=figsize)
        if title and self.label is not None: ax.set_title(self.label)
        ax.set_axis_off()
        self.graphparams['pos'] = nx.nx_pydot.graphviz_layout(self.G, prog='dot')
        for i, node in enumerate(self.G.nodes):
            nx.draw_networkx_nodes(self.G,
                                   pos=self.graphparams['pos'],
                                   nodelist=[node],
                                   with_labels=True,
                                   node_size=node_size,
                                   node_color=nodecolors[i],
                                   edgecolors=edgecolors[i],
                                   node_shape=nodeshapes[i],
                                   linewidths=lw,
                                   font_size=font_size)
        nx.draw_networkx_labels(self.G,
                                nodelist=[node],
                                pos=self.graphparams['pos'],
                                font_color=fontcolors[i])
        nx.draw_networkx_edges(self.G,
                               pos=self.graphparams['pos'],
                               width=lw,
                               arrowstyle='-|>',
                               node_size=node_size,
                               arrows=True,
                               arrowsize=arrowsize)
        nx.draw_networkx_edge_labels(self.G,
                                     pos=self.graphparams['pos'],
                                     edge_labels=self.graphparams['edgelabels'],
                                     label_pos=0.5,
                                     rotate=True)
        if outfile is None:
            plt.show()
        else:
            if outfiletype == 'png':
                plt.savefig(outfile+'.png', dpi=350, bbox_inches='tight')
            else:
                plt.savefig(outfile+'.'+outfiletype, bbox_inches='tight')

    def plot_spectral_properties(self, figsize=None, outfile=None, outfiletype='pdf'):
        """ Creates a set of subplots depicting the graph Laplacian and its spectral decomposition. """
        if figsize is None:
            figsize=(10, 4)

        fig, ax = plt.subplots(ncols=3, figsize=figsize)
        ax[0].set_title('Laplacian Matrix')
        ax[0].set_axis_off()
        ax[0].imshow(self.laplacian, cmap=self.cmap)
        ax[1].set_title('Laplacian Spectrum')
        ax[1].set_xlabel('Component')
        ax[1].set_xticklabels(np.round(np.arange(self.laplacian_spectrum.size), 0).astype(np.int))
        ax[1].set_ylabel('Singular value')
        ax[1].plot(np.arange(self.laplacian_spectrum.size),
                             self.laplacian_spectrum,
                             c='k')
        ax[1].scatter(np.arange(self.laplacian_spectrum.size),
                      self.laplacian_spectrum,
                      c='k')
        ax[2].set_title('Spectral Embedding')
        ax[2].set_xlabel('PC1')
        ax[2].set_ylabel('PC2')
        ax[2].scatter(self.laplacian_eigenvectors[:,0],
                      self.laplacian_eigenvectors[:,1],
                      c='k')
        plt.tight_layout()

        if outfile is None:
            plt.show()
        else:
            if outfiletype == 'png':
                plt.savefig(outfile+'.png', dpi=350, bbox_inches='tight')
            else:
                plt.savefig(outfile+'.'+outfiletype, bbox_inches='tight')

    def plot_action_outcome_probabilities(self,
                                          figsize=None,
                                          outfile=None,
                                          outfiletype='pdf',
                                          cmap='Greys_r'):
                """ Plots the probabilities of different outcomes given actions.

                Each plot is a heatmap for a starting state showing the transition probabilities for each action-outcome pair within that state.
                """
                if figsize is None:
                    figsize = (10, 10)


                fig, ax = plt.subplots(ncols=self.nstates-self.noutcomes, figsize=figsize)
                for i in range(self.nstates-self.noutcomes):
                    #if np.sum(self.T[:,:,i])>0:
                    if i == 0:
                        ax[i].set_yticks(np.arange(self.nactions))
                        ax[i].set_yticklabels(self.action_labels)
                    else:
                        ax[i].set_yticks([])
                        ax[i].set_yticklabels([])
                    ax[i].imshow(self.T[:,-self.noutcomes:,i], cmap=cmap)
                    ax[i].set_title(self.state_labels[i])
                    ax[i].set_xticks(np.arange(self.noutcomes))
                    ax[i].set_xticklabels(self.state_labels[-self.noutcomes:])



                if outfile is None:
                    plt.show()
                else:
                    if outfiletype == 'png':
                        plt.savefig(outfile+'.png', dpi=350, bbox_inches='tight')
                    else:
                        plt.savefig(outfile+'.'+outfiletype, bbox_inches='tight')

#===============================================================================
#   SIMPLE FUNCTIONS TO RUN ENVIRONMENTS
#===============================================================================

def generate_behavioural_data(environment, Agent, nsubjects, ntrials):
    """
    A function for flexibly simulating data for different task/agent combos.

    Arguments:

        environment: An instantiated `Graph` object representing the task being tested
        Agent: A `fitr.agents.Agent` object representing the agent being evaluated
        nsubjects: An `int` number of subjects to simulate
        ntrials: An `int` number of trials to simulate

    Returns:

        A `BehaviouralData` object containing all data simulated from the current run.

    Examples:

        ```python
        from fitr.agents import RWSoftmaxAgent
        from fitr.environments import TwoArmedBandit

        ```
    """
    for i in range(nsubjects):
        agent = Agent(environment)
        subject_data = agent.generate_data(ntrials)
        if i == 0:
            data = subject_data
        else:
            data = merge_behavioural_data([data, subject_data])
    return data

#===============================================================================
#   REWARD REFLECTION
#===============================================================================

def reward_reflection(x, lb, ub):
    """ Imposes reflective boundaries on drifting reward functions.

    Denoting the lower bound by $l$ and the upper bound by $u$, this is computed according to the following formula:

    $$
    \max \Big\{\min \big\{\mathbf x, \max \big\{2u-\mathbf x, l \big\} \big\}, \min \big\{2l-\mathbf x, u \big\} \Big\}
    $$

    Arguments:

        x: An `ndarray((n,))` vector of values
        lb: A `float` depicting the lower bound for the rewards
        ub: A `float` depicting the upper bound for rewards

    Return:

        An updated reward vector `ndarray((n,))`
    """
    q = np.minimum(2*lb-x, ub)
    h = np.maximum(2*ub-x, lb)
    g = np.minimum(x, h)
    return np.maximum(g, q)

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
#   SPECIFIC TASKS
#===============================================================================

class TwoArmedBandit(Graph):
    """ Two armed bandit just as a tester """
    def __init__(self):
        T = np.zeros((2, 3, 3))
        T[0,1,0] = 0.8      # These end up being reward probabilities
        T[0,2,0] = 0.2      #  because states are deterministically rewarded
        T[1,1,0] = 0.2
        T[1,2,0] = 0.8

        p_start = np.array([1.,0.,0.])
        R = np.array([0.,0.,1.])
        xend = np.array([0.,1.,1.])
        super().__init__(T,R,xend,p_start)

class OrthogonalGoNoGo(Graph):
    """
    The orthogonal GoNogo task from Guitart-Masip et al. (2012)
    """
    def __init__(self):
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
                         action_labels=alabs, label=taskname)

class TwoStep(Graph):
    """ An implementation of the Two-Step Task from Daw et al. (2011).

    Arguments:

        mu: `float` identifying the drift of the reward-determining Gaussian random walks
        sd: `float` identifying the standard deviation of the reward-determining Gaussian random walks

    """
    def __init__(self, mu=0, sd=0.025):
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
                         label=taskname, edge_labels=edge_labels)

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

class ReverseTwoStep(Graph):
    """
    From Kool & Gershman 2016.
    """
    def __init__(self, mu=0, sd=2, path_corr=0, reward_lb=-4, reward_ub=5):
        T = np.zeros((2,4,4))
        T[0,2,0] = 1.
        T[0,3,0] = 0.
        T[1,2,0] = 0.
        T[1,3,0] = 1.

        T[0,2,1] = 0.
        T[0,3,1] = 1.
        T[1,2,1] = 1.
        T[1,3,1] = 0.

        p_start    = np.array([0.5,0.5,0,0])
        R          = np.array([0,0,0.3,0.7])
        end_states = np.array([0,0,1,1])
        super().__init__(T,R,end_states,p_start,f_reward=self.f_reward)

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
        self.R[2:] = self.R[2:] + self.mvn.rvs()
        self.R[2:] = reward_reflection(self.R[2:], self.reward_lb, self.reward_ub)
        self.reward_hx = np.vstack((self.reward_hx, self.R))
        return np.einsum('s,s->',self.R,x)

class ReverseTwoStepWithAvoidance(Graph):
    """
    Modified from Kool, Cushman, & Gershman 2016.
    """
    def __init__(self, mu=0, sd=2, path_corr=0., reward_lb=-4, reward_ub=5):
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
                         state_labels=slabs, action_labels=alabs)

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

class MouthTask(Graph):
    """ The Pizzagalli reward sensitivity signal-detection task """
    def __init__(self):
        T = np.zeros((2, 4, 4))
        #[u,x',x]
        T[0,2,0] = 0.75
        T[0,3,0] = 0.3
        T[1,2,0] = 0.3
        T[1,3,0] = 0.75

        T[0,2,1] = 0.0
        T[0,3,1] = 0.75
        T[1,2,1] = 0.0
        T[1,3,1] = 0.75

        p_start = np.array([0.5,0.5,0.,0.])
        R = np.array([0.,0.,1.,0])
        xend = np.array([0.,0.,1.,1.])
        slabs = [r'$s_{0}$', r'$s_{1}$', r'Win', r'Nil']
        alabs = [r'$a_{0}$', r'$a_{1}$']
        taskname = 'Reward Signal Detection Task'
        super().__init__(T,R,xend,p_start,state_labels=slabs,
                         action_labels=alabs, label=taskname)
        self.Ti[0,2,1] = 1.
        self.Ti[1,2,1] = 1.

class IGT(Graph):
    """ Iowa Gambling Task """
    def __init__(self):
        T = np.zeros((4, 5, 5))
        T[0,1,0] = 0.5
        T[0,3,0] = 0.5
        T[1,1,0] = 0.5
        T[1,3,0] = 0.5
        T[2,2,0] = 0.5
        T[2,4,0] = 0.5
        T[3,2,0] = 0.5
        T[3,4,0] = 0.5

        p_start = np.array([1., 0., 0., 0.])
        R = np.array([0., 100., 50., -250., -50.])
        xend = np.array([0., 1., 1., 1., 1.])
        slabs = [r'$s_0$', '+$100', '+$50', '-$250', '-$50']
        alabs = [r'$a_A$', r'$a_B$', r'$a_C$', r'$a_D$']
        taskname = 'Iowa Gambling Task'
        super().__init__(T, R, xend, p_start, state_labels=slabs,
                         action_labels=alabs, label=taskname)

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
                 drift_sd=2.):
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
        super().__init__(T,R,xend,p_start)

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

    def f_reward(self, R, x):
        if self.reward_drift == 'on':
            perturbation = self.mvn.rvs()
        elif self.reward_drift == 'off':
            perturbation = np.zeros(self.noutcomes)
        self.R[-self.noutcomes:] = self.R[-self.noutcomes:] + perturbation
        self.R[-self.noutcomes:] = reward_reflection(self.R[-self.noutcomes:], self.reward_lb, self.reward_ub)
        self.reward_hx = np.vstack((self.reward_hx, self.R))
        return np.einsum('s,s->',self.R,x)
