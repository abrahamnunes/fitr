# -*- coding: utf-8 -*-
import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

    def set_seed(self, seed=None):
        """ Allows user to specify a seed for the pseudorandom number generator.

        Arguments:

            seed: `int`. Seed value. Default is `None`, which results in a default random state object. If user enters a non-integer value, the default random state object will still be used and no error will be thrown!

        """
        if type(seed) is np.int:
            self.rng = np.random.RandomState(seed)

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
