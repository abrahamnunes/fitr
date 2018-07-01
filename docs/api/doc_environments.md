# `fitr.environments`

Functions to synthesize data from behavioural tasks.




## Graph

```python
fitr.environments.Graph()
```

Base object that defines a reinforcement learning task.

#### Definitions

- $\mathbf x \in \mathcal X$ be a one-hot state vector, where $|\mathcal X|=n_x$
- $\mathbf u \in \mathcal U$ be a one-hot action vector, where $|\mathcal U|=n_u$
- $\mathsf T = p(\mathbf x_{t+1}|\mathbf x_t, \mathbf u_t)$ be a transition tensor
- $p(\mathbf x)$ be a distribution over starting states
- $\mathcal J: \mathcal X \to \mathcal R$, where $\mathcal R \subseteq \mathbb R$ be a reward function

Arguments:

- **T**: Transition tensor
- **R**: Vector of rewards for each state such that scalar reward $r_t = \mathbf r^   op \mathbf x$
- **end_states**: A vector $\{0, 1\}^{n_x}$ identifying which states terminate a trial (aka episode)
- **p_start**: Initial state distribution
- **label**: A string identifying a name for the task
- **state_labels**: A list or array of strings labeling the different states (for plotting purposes)
- **action_labels**: A list or array of strings labeling the different actions (for plotting purposes)
- **rng**: `np.random.RandomState` object
- **f_reward**: A function whose first argument is a vector of rewards for each state, and whose second argument is a state vector, and whose output is a scalar reward
- **cmap**: Matplotlib colormap for plotting.

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

---




### Graph.adjacency_matrix_decomposition

```python
fitr.environments.adjacency_matrix_decomposition(self)
```

Singular value decomposition of the graph adjacency matrix 

---




### Graph.get_graph_depth

```python
fitr.environments.get_graph_depth(self)
```

Returns the depth of the task graph.

Calculated as the depth from `START` (pre-initial state) to `END` (which absorbs trial from all terminal states), minus 2 to account for the `START->node` & `node->END` transitions.

Returns:

An `int` identifying the depth of the current graph for a single trial of the task

---




### Graph.laplacian_matrix_decomposition

```python
fitr.environments.laplacian_matrix_decomposition(self)
```

Singular value decomposition of the graph Laplacian 

---




### Graph.make_action_labels

```python
fitr.environments.make_action_labels(self)
```

Creates labels for the actions (for plotting) if none provided 

---




### Graph.make_digraph

```python
fitr.environments.make_digraph(self)
```

Creates a `networkx` `DiGraph` object from the transition tensor for the purpose of plotting and some other analyses. 

---




### Graph.make_state_labels

```python
fitr.environments.make_state_labels(self)
```

Creates labels for the states (for plotting) if none provided 

---




### Graph.make_undirected_graph

```python
fitr.environments.make_undirected_graph(self)
```

Converts the DiGraph to undirected and computes some stats 

---




### Graph.observation

```python
fitr.environments.observation(self)
```

Samples an initial state from the start-state distribution $p(\mathbf x)$

$$
\mathbf x_0 \sim p(\mathbf x)
$$

Returns:

A one-hot vector `ndarray((nstates,))` indicating the starting state.

Examples:

```python
x = env.observation()
```

---




### Graph.plot_action_outcome_probabilities

```python
fitr.environments.plot_action_outcome_probabilities(self, figsize=None, outfile=None, outfiletype='pdf', cmap='Greys_r')
```

Plots the probabilities of different outcomes given actions.

Each plot is a heatmap for a starting state showing the transition probabilities for each action-outcome pair within that state.

---




### Graph.plot_graph

```python
fitr.environments.plot_graph(self, figsize=None, node_size=2000, arrowsize=20, lw=1.5, font_size=12, title=False, outfile=None, outfiletype='pdf')
```

Plots the directed graph of the task 

---




### Graph.plot_spectral_properties

```python
fitr.environments.plot_spectral_properties(self, figsize=None, outfile=None, outfiletype='pdf')
```

Creates a set of subplots depicting the graph Laplacian and its spectral decomposition. 

---




### Graph.random_action

```python
fitr.environments.random_action(self)
```

Samples a random one-hot action vector uniformly over the action space.

Useful for testing that your environment works, without having to create an agent.

$$
\mathbf u \sim \mathrm{Multinomial}\Big(1, \mathbf p=\{p_i = \frac{1}{|\mathcal U|}\}_{i=1}^{|\mathcal U|}\Big)
$$

Returns:

A one-hot action vector of type `ndarray((nactions,))`

Examples:

```python
u = env.random_action()
```

---




### Graph.step

```python
fitr.environments.step(self, action)
```

Executes a state transition in the environment.

Arguments:

action : A one-hot vector of type `ndarray((naction,))` indicating the action selected at the current state.

Returns:

A 3-tuple representing the next state (`ndarray((noutcomes,))`), scalar reward, and whether the current step terminates a trial (`bool`).

Raises:

`RuntimeError` if `env.observation()` not called after a previous `env.step(...)` call yielded a terminal state.

---



## TwoArmedBandit

```python
fitr.environments.TwoArmedBandit()
```

Two armed bandit just as a tester 

---




### TwoArmedBandit.adjacency_matrix_decomposition

```python
fitr.environments.adjacency_matrix_decomposition(self)
```

Singular value decomposition of the graph adjacency matrix 

---




### TwoArmedBandit.get_graph_depth

```python
fitr.environments.get_graph_depth(self)
```

Returns the depth of the task graph.

Calculated as the depth from `START` (pre-initial state) to `END` (which absorbs trial from all terminal states), minus 2 to account for the `START->node` & `node->END` transitions.

Returns:

An `int` identifying the depth of the current graph for a single trial of the task

---




### TwoArmedBandit.laplacian_matrix_decomposition

```python
fitr.environments.laplacian_matrix_decomposition(self)
```

Singular value decomposition of the graph Laplacian 

---




### TwoArmedBandit.make_action_labels

```python
fitr.environments.make_action_labels(self)
```

Creates labels for the actions (for plotting) if none provided 

---




### TwoArmedBandit.make_digraph

```python
fitr.environments.make_digraph(self)
```

Creates a `networkx` `DiGraph` object from the transition tensor for the purpose of plotting and some other analyses. 

---




### TwoArmedBandit.make_state_labels

```python
fitr.environments.make_state_labels(self)
```

Creates labels for the states (for plotting) if none provided 

---




### TwoArmedBandit.make_undirected_graph

```python
fitr.environments.make_undirected_graph(self)
```

Converts the DiGraph to undirected and computes some stats 

---




### TwoArmedBandit.observation

```python
fitr.environments.observation(self)
```

Samples an initial state from the start-state distribution $p(\mathbf x)$

$$
\mathbf x_0 \sim p(\mathbf x)
$$

Returns:

A one-hot vector `ndarray((nstates,))` indicating the starting state.

Examples:

```python
x = env.observation()
```

---




### TwoArmedBandit.plot_action_outcome_probabilities

```python
fitr.environments.plot_action_outcome_probabilities(self, figsize=None, outfile=None, outfiletype='pdf', cmap='Greys_r')
```

Plots the probabilities of different outcomes given actions.

Each plot is a heatmap for a starting state showing the transition probabilities for each action-outcome pair within that state.

---




### TwoArmedBandit.plot_graph

```python
fitr.environments.plot_graph(self, figsize=None, node_size=2000, arrowsize=20, lw=1.5, font_size=12, title=False, outfile=None, outfiletype='pdf')
```

Plots the directed graph of the task 

---




### TwoArmedBandit.plot_spectral_properties

```python
fitr.environments.plot_spectral_properties(self, figsize=None, outfile=None, outfiletype='pdf')
```

Creates a set of subplots depicting the graph Laplacian and its spectral decomposition. 

---




### TwoArmedBandit.random_action

```python
fitr.environments.random_action(self)
```

Samples a random one-hot action vector uniformly over the action space.

Useful for testing that your environment works, without having to create an agent.

$$
\mathbf u \sim \mathrm{Multinomial}\Big(1, \mathbf p=\{p_i = \frac{1}{|\mathcal U|}\}_{i=1}^{|\mathcal U|}\Big)
$$

Returns:

A one-hot action vector of type `ndarray((nactions,))`

Examples:

```python
u = env.random_action()
```

---




### TwoArmedBandit.step

```python
fitr.environments.step(self, action)
```

Executes a state transition in the environment.

Arguments:

action : A one-hot vector of type `ndarray((naction,))` indicating the action selected at the current state.

Returns:

A 3-tuple representing the next state (`ndarray((noutcomes,))`), scalar reward, and whether the current step terminates a trial (`bool`).

Raises:

`RuntimeError` if `env.observation()` not called after a previous `env.step(...)` call yielded a terminal state.

---



## OrthogonalGoNoGo

```python
fitr.environments.OrthogonalGoNoGo()
```

The orthogonal GoNogo task from Guitart-Masip et al. (2012)

---




### OrthogonalGoNoGo.adjacency_matrix_decomposition

```python
fitr.environments.adjacency_matrix_decomposition(self)
```

Singular value decomposition of the graph adjacency matrix 

---




### OrthogonalGoNoGo.get_graph_depth

```python
fitr.environments.get_graph_depth(self)
```

Returns the depth of the task graph.

Calculated as the depth from `START` (pre-initial state) to `END` (which absorbs trial from all terminal states), minus 2 to account for the `START->node` & `node->END` transitions.

Returns:

An `int` identifying the depth of the current graph for a single trial of the task

---




### OrthogonalGoNoGo.laplacian_matrix_decomposition

```python
fitr.environments.laplacian_matrix_decomposition(self)
```

Singular value decomposition of the graph Laplacian 

---




### OrthogonalGoNoGo.make_action_labels

```python
fitr.environments.make_action_labels(self)
```

Creates labels for the actions (for plotting) if none provided 

---




### OrthogonalGoNoGo.make_digraph

```python
fitr.environments.make_digraph(self)
```

Creates a `networkx` `DiGraph` object from the transition tensor for the purpose of plotting and some other analyses. 

---




### OrthogonalGoNoGo.make_state_labels

```python
fitr.environments.make_state_labels(self)
```

Creates labels for the states (for plotting) if none provided 

---




### OrthogonalGoNoGo.make_undirected_graph

```python
fitr.environments.make_undirected_graph(self)
```

Converts the DiGraph to undirected and computes some stats 

---




### OrthogonalGoNoGo.observation

```python
fitr.environments.observation(self)
```

Samples an initial state from the start-state distribution $p(\mathbf x)$

$$
\mathbf x_0 \sim p(\mathbf x)
$$

Returns:

A one-hot vector `ndarray((nstates,))` indicating the starting state.

Examples:

```python
x = env.observation()
```

---




### OrthogonalGoNoGo.plot_action_outcome_probabilities

```python
fitr.environments.plot_action_outcome_probabilities(self, figsize=None, outfile=None, outfiletype='pdf', cmap='Greys_r')
```

Plots the probabilities of different outcomes given actions.

Each plot is a heatmap for a starting state showing the transition probabilities for each action-outcome pair within that state.

---




### OrthogonalGoNoGo.plot_graph

```python
fitr.environments.plot_graph(self, figsize=None, node_size=2000, arrowsize=20, lw=1.5, font_size=12, title=False, outfile=None, outfiletype='pdf')
```

Plots the directed graph of the task 

---




### OrthogonalGoNoGo.plot_spectral_properties

```python
fitr.environments.plot_spectral_properties(self, figsize=None, outfile=None, outfiletype='pdf')
```

Creates a set of subplots depicting the graph Laplacian and its spectral decomposition. 

---




### OrthogonalGoNoGo.random_action

```python
fitr.environments.random_action(self)
```

Samples a random one-hot action vector uniformly over the action space.

Useful for testing that your environment works, without having to create an agent.

$$
\mathbf u \sim \mathrm{Multinomial}\Big(1, \mathbf p=\{p_i = \frac{1}{|\mathcal U|}\}_{i=1}^{|\mathcal U|}\Big)
$$

Returns:

A one-hot action vector of type `ndarray((nactions,))`

Examples:

```python
u = env.random_action()
```

---




### OrthogonalGoNoGo.step

```python
fitr.environments.step(self, action)
```

Executes a state transition in the environment.

Arguments:

action : A one-hot vector of type `ndarray((naction,))` indicating the action selected at the current state.

Returns:

A 3-tuple representing the next state (`ndarray((noutcomes,))`), scalar reward, and whether the current step terminates a trial (`bool`).

Raises:

`RuntimeError` if `env.observation()` not called after a previous `env.step(...)` call yielded a terminal state.

---



## TwoStep

```python
fitr.environments.TwoStep()
```

An implementation of the Two-Step Task from Daw et al. (2011).

Arguments:

- **mu**: `float` identifying the drift of the reward-determining Gaussian random walks
- **sd**: `float` identifying the standard deviation of the reward-determining Gaussian random walks

---




### TwoStep.adjacency_matrix_decomposition

```python
fitr.environments.adjacency_matrix_decomposition(self)
```

Singular value decomposition of the graph adjacency matrix 

---




### TwoStep.f_reward

```python
fitr.environments.f_reward(self, R, x)
```



---




### TwoStep.get_graph_depth

```python
fitr.environments.get_graph_depth(self)
```

Returns the depth of the task graph.

Calculated as the depth from `START` (pre-initial state) to `END` (which absorbs trial from all terminal states), minus 2 to account for the `START->node` & `node->END` transitions.

Returns:

An `int` identifying the depth of the current graph for a single trial of the task

---




### TwoStep.laplacian_matrix_decomposition

```python
fitr.environments.laplacian_matrix_decomposition(self)
```

Singular value decomposition of the graph Laplacian 

---




### TwoStep.make_action_labels

```python
fitr.environments.make_action_labels(self)
```

Creates labels for the actions (for plotting) if none provided 

---




### TwoStep.make_digraph

```python
fitr.environments.make_digraph(self)
```

Creates a `networkx` `DiGraph` object from the transition tensor for the purpose of plotting and some other analyses. 

---




### TwoStep.make_state_labels

```python
fitr.environments.make_state_labels(self)
```

Creates labels for the states (for plotting) if none provided 

---




### TwoStep.make_undirected_graph

```python
fitr.environments.make_undirected_graph(self)
```

Converts the DiGraph to undirected and computes some stats 

---




### TwoStep.observation

```python
fitr.environments.observation(self)
```

Samples an initial state from the start-state distribution $p(\mathbf x)$

$$
\mathbf x_0 \sim p(\mathbf x)
$$

Returns:

A one-hot vector `ndarray((nstates,))` indicating the starting state.

Examples:

```python
x = env.observation()
```

---




### TwoStep.plot_action_outcome_probabilities

```python
fitr.environments.plot_action_outcome_probabilities(self, figsize=None, outfile=None, outfiletype='pdf', cmap='Greys_r')
```

Plots the probabilities of different outcomes given actions.

Each plot is a heatmap for a starting state showing the transition probabilities for each action-outcome pair within that state.

---




### TwoStep.plot_graph

```python
fitr.environments.plot_graph(self, figsize=None, node_size=2000, arrowsize=20, lw=1.5, font_size=12, title=False, outfile=None, outfiletype='pdf')
```

Plots the directed graph of the task 

---




### TwoStep.plot_reward_paths

```python
fitr.environments.plot_reward_paths(self, outfile=None, outfiletype='pdf', figsize=None)
```



---




### TwoStep.plot_spectral_properties

```python
fitr.environments.plot_spectral_properties(self, figsize=None, outfile=None, outfiletype='pdf')
```

Creates a set of subplots depicting the graph Laplacian and its spectral decomposition. 

---




### TwoStep.random_action

```python
fitr.environments.random_action(self)
```

Samples a random one-hot action vector uniformly over the action space.

Useful for testing that your environment works, without having to create an agent.

$$
\mathbf u \sim \mathrm{Multinomial}\Big(1, \mathbf p=\{p_i = \frac{1}{|\mathcal U|}\}_{i=1}^{|\mathcal U|}\Big)
$$

Returns:

A one-hot action vector of type `ndarray((nactions,))`

Examples:

```python
u = env.random_action()
```

---




### TwoStep.step

```python
fitr.environments.step(self, action)
```

Executes a state transition in the environment.

Arguments:

action : A one-hot vector of type `ndarray((naction,))` indicating the action selected at the current state.

Returns:

A 3-tuple representing the next state (`ndarray((noutcomes,))`), scalar reward, and whether the current step terminates a trial (`bool`).

Raises:

`RuntimeError` if `env.observation()` not called after a previous `env.step(...)` call yielded a terminal state.

---



## ReverseTwoStep

```python
fitr.environments.ReverseTwoStep()
```

From Kool & Gershman 2016.

---




### ReverseTwoStep.adjacency_matrix_decomposition

```python
fitr.environments.adjacency_matrix_decomposition(self)
```

Singular value decomposition of the graph adjacency matrix 

---




### ReverseTwoStep.f_reward

```python
fitr.environments.f_reward(self, R, x)
```



---




### ReverseTwoStep.get_graph_depth

```python
fitr.environments.get_graph_depth(self)
```

Returns the depth of the task graph.

Calculated as the depth from `START` (pre-initial state) to `END` (which absorbs trial from all terminal states), minus 2 to account for the `START->node` & `node->END` transitions.

Returns:

An `int` identifying the depth of the current graph for a single trial of the task

---




### ReverseTwoStep.laplacian_matrix_decomposition

```python
fitr.environments.laplacian_matrix_decomposition(self)
```

Singular value decomposition of the graph Laplacian 

---




### ReverseTwoStep.make_action_labels

```python
fitr.environments.make_action_labels(self)
```

Creates labels for the actions (for plotting) if none provided 

---




### ReverseTwoStep.make_digraph

```python
fitr.environments.make_digraph(self)
```

Creates a `networkx` `DiGraph` object from the transition tensor for the purpose of plotting and some other analyses. 

---




### ReverseTwoStep.make_state_labels

```python
fitr.environments.make_state_labels(self)
```

Creates labels for the states (for plotting) if none provided 

---




### ReverseTwoStep.make_undirected_graph

```python
fitr.environments.make_undirected_graph(self)
```

Converts the DiGraph to undirected and computes some stats 

---




### ReverseTwoStep.observation

```python
fitr.environments.observation(self)
```

Samples an initial state from the start-state distribution $p(\mathbf x)$

$$
\mathbf x_0 \sim p(\mathbf x)
$$

Returns:

A one-hot vector `ndarray((nstates,))` indicating the starting state.

Examples:

```python
x = env.observation()
```

---




### ReverseTwoStep.plot_action_outcome_probabilities

```python
fitr.environments.plot_action_outcome_probabilities(self, figsize=None, outfile=None, outfiletype='pdf', cmap='Greys_r')
```

Plots the probabilities of different outcomes given actions.

Each plot is a heatmap for a starting state showing the transition probabilities for each action-outcome pair within that state.

---




### ReverseTwoStep.plot_graph

```python
fitr.environments.plot_graph(self, figsize=None, node_size=2000, arrowsize=20, lw=1.5, font_size=12, title=False, outfile=None, outfiletype='pdf')
```

Plots the directed graph of the task 

---




### ReverseTwoStep.plot_spectral_properties

```python
fitr.environments.plot_spectral_properties(self, figsize=None, outfile=None, outfiletype='pdf')
```

Creates a set of subplots depicting the graph Laplacian and its spectral decomposition. 

---




### ReverseTwoStep.random_action

```python
fitr.environments.random_action(self)
```

Samples a random one-hot action vector uniformly over the action space.

Useful for testing that your environment works, without having to create an agent.

$$
\mathbf u \sim \mathrm{Multinomial}\Big(1, \mathbf p=\{p_i = \frac{1}{|\mathcal U|}\}_{i=1}^{|\mathcal U|}\Big)
$$

Returns:

A one-hot action vector of type `ndarray((nactions,))`

Examples:

```python
u = env.random_action()
```

---




### ReverseTwoStep.step

```python
fitr.environments.step(self, action)
```

Executes a state transition in the environment.

Arguments:

action : A one-hot vector of type `ndarray((naction,))` indicating the action selected at the current state.

Returns:

A 3-tuple representing the next state (`ndarray((noutcomes,))`), scalar reward, and whether the current step terminates a trial (`bool`).

Raises:

`RuntimeError` if `env.observation()` not called after a previous `env.step(...)` call yielded a terminal state.

---



## RandomContextualBandit

```python
fitr.environments.RandomContextualBandit()
```

Generates a random bandit task

Arguments:

- **nactions**: Number of actions
- **noutcomes**: Number of outcomes
- **nstates**: Number of contexts
- **min_actions_per_context**: Different contexts may have more or fewer actions than others (never more than `nactions`). This variable describes the minimum number of actions allowed in a context.
- **alpha**:
- **alpha_start**:
- **shift_flip**:
- **reward_lb**: Lower bound for drifting rewards
- **reward_ub**: Upper bound for drifting rewards
- **reward_drift**: Values (`on` or `off`) determining whether rewards are allowed to drift
- **drift_mu**: Mean of the Gaussian random walk determining reward
- **drift_sd**: Standard deviation of Gaussian random walk determining reward

---




### RandomContextualBandit.adjacency_matrix_decomposition

```python
fitr.environments.adjacency_matrix_decomposition(self)
```

Singular value decomposition of the graph adjacency matrix 

---




### RandomContextualBandit.f_reward

```python
fitr.environments.f_reward(self, R, x)
```



---




### RandomContextualBandit.get_graph_depth

```python
fitr.environments.get_graph_depth(self)
```

Returns the depth of the task graph.

Calculated as the depth from `START` (pre-initial state) to `END` (which absorbs trial from all terminal states), minus 2 to account for the `START->node` & `node->END` transitions.

Returns:

An `int` identifying the depth of the current graph for a single trial of the task

---




### RandomContextualBandit.laplacian_matrix_decomposition

```python
fitr.environments.laplacian_matrix_decomposition(self)
```

Singular value decomposition of the graph Laplacian 

---




### RandomContextualBandit.make_action_labels

```python
fitr.environments.make_action_labels(self)
```

Creates labels for the actions (for plotting) if none provided 

---




### RandomContextualBandit.make_digraph

```python
fitr.environments.make_digraph(self)
```

Creates a `networkx` `DiGraph` object from the transition tensor for the purpose of plotting and some other analyses. 

---




### RandomContextualBandit.make_state_labels

```python
fitr.environments.make_state_labels(self)
```

Creates labels for the states (for plotting) if none provided 

---




### RandomContextualBandit.make_undirected_graph

```python
fitr.environments.make_undirected_graph(self)
```

Converts the DiGraph to undirected and computes some stats 

---




### RandomContextualBandit.observation

```python
fitr.environments.observation(self)
```

Samples an initial state from the start-state distribution $p(\mathbf x)$

$$
\mathbf x_0 \sim p(\mathbf x)
$$

Returns:

A one-hot vector `ndarray((nstates,))` indicating the starting state.

Examples:

```python
x = env.observation()
```

---




### RandomContextualBandit.plot_action_outcome_probabilities

```python
fitr.environments.plot_action_outcome_probabilities(self, figsize=None, outfile=None, outfiletype='pdf', cmap='Greys_r')
```

Plots the probabilities of different outcomes given actions.

Each plot is a heatmap for a starting state showing the transition probabilities for each action-outcome pair within that state.

---




### RandomContextualBandit.plot_graph

```python
fitr.environments.plot_graph(self, figsize=None, node_size=2000, arrowsize=20, lw=1.5, font_size=12, title=False, outfile=None, outfiletype='pdf')
```

Plots the directed graph of the task 

---




### RandomContextualBandit.plot_spectral_properties

```python
fitr.environments.plot_spectral_properties(self, figsize=None, outfile=None, outfiletype='pdf')
```

Creates a set of subplots depicting the graph Laplacian and its spectral decomposition. 

---




### RandomContextualBandit.random_action

```python
fitr.environments.random_action(self)
```

Samples a random one-hot action vector uniformly over the action space.

Useful for testing that your environment works, without having to create an agent.

$$
\mathbf u \sim \mathrm{Multinomial}\Big(1, \mathbf p=\{p_i = \frac{1}{|\mathcal U|}\}_{i=1}^{|\mathcal U|}\Big)
$$

Returns:

A one-hot action vector of type `ndarray((nactions,))`

Examples:

```python
u = env.random_action()
```

---




### RandomContextualBandit.step

```python
fitr.environments.step(self, action)
```

Executes a state transition in the environment.

Arguments:

action : A one-hot vector of type `ndarray((naction,))` indicating the action selected at the current state.

Returns:

A 3-tuple representing the next state (`ndarray((noutcomes,))`), scalar reward, and whether the current step terminates a trial (`bool`).

Raises:

`RuntimeError` if `env.observation()` not called after a previous `env.step(...)` call yielded a terminal state.

---


