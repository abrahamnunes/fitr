# `fitr.agents`

A modular way to build and test reinforcement learning agents.

There are three main submodules:

- `fitr.agents.policies`: which describe a class of functions essentially representing $f:\mathcal X \to \mathcal U$
- `fitr.agents.value_functions`: which describe a class of functions essentially representing $\mathcal V: \mathcal X \to \mathbb R$ and/or $\mathcal Q: \mathcal Q \times \mathcal U \to \mathbb R$
- `fitr.agents.agents`: classes of agents that are combinations of policies and value functions, along with some convenience functions for generating data from `fitr.environments.Graph` environments.



## SoftmaxPolicy

```python
fitr.agents.policies.SoftmaxPolicy()
```

Action selection by sampling from a multinomial whose parameters are given by a softmax.

Action sampling is

$$
\mathbf u \sim \mathrm{Multinomial}(1, \mathbf p=\varsigma(\mathbf v)).
$$

Parameters of that distribution are

$$
p(\mathbf u|\mathbf v) = \varsigma(\mathbf v) = \frac{e^{\beta \mathbf v}}{\sum_{i}^{|\mathbf v|} e^{\beta v_i}}.
$$

Arguments:

- **inverse_softmax_temp**: Inverse softmax temperature $\beta$
- **rng**: `np.random.RandomState` object

---




### SoftmaxPolicy.action_prob

```python
fitr.agents.policies.action_prob(self, x)
```

Computes the softmax 

---




### SoftmaxPolicy.log_prob

```python
fitr.agents.policies.log_prob(self, x)
```

Computes the log-probability of an action $\mathbf u$

$$
\log p(\mathbf u|\mathbf v) = \beta \mathbf v - \log \sum_{v_i} e^{\beta \mathbf v_i}
$$

Arguments:

- **x**: State vector of type `ndarray((nstates,))`

Returns:

Scalar log-probability

---




### SoftmaxPolicy.sample

```python
fitr.agents.policies.sample(self, x)
```

Samples from the action distribution 

---



## StickySoftmaxPolicy

```python
fitr.agents.policies.StickySoftmaxPolicy()
```

Action selection by sampling from a multinomial whose parameters are given by a softmax, but with accounting for the tendency to perseverate (i.e. choosing the previously used action without considering its value).

Let $\mathbf u_{t-1} = (u_{t-1}^{(i)})_{i=1}^{|\mathcal U|}$ be a one hot vector representing the action taken at the last step, and $\beta^\rho$ be an inverse softmax temperature for the influence of this last action.

Action sampling is thus:

$$
\mathbf u \sim \mathrm{Multinomial}(1, \mathbf p=\varsigma(\mathbf v, \mathbf u_{t-1})).
$$

Parameters of that distribution are

$$
p(\mathbf u|\mathbf v, \mathbf u_{t-1}) = \varsigma(\mathbf v, \mathbf u_{t-1}) = \frac{e^{\beta \mathbf v + \beta^\rho \mathbf u_{t-1}}}{\sum_{i}^{|\mathbf v|} e^{\beta v_i + \beta^\rho u_{t-1}^{(i)}}}.
$$

Arguments:

- **inverse_softmax_temp**: Inverse softmax temperature $\beta$
- **perseveration**: Inverse softmax temperature $\beta^\rho$ capturing the tendency to repeat the last action taken.
- **rng**: `np.random.RandomState` object

---




### StickySoftmaxPolicy.action_prob

```python
fitr.agents.policies.action_prob(self, x)
```

Computes the softmax

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

`ndarray((nstates,))` vector of action probabilities

---




### StickySoftmaxPolicy.log_prob

```python
fitr.agents.policies.log_prob(self, x)
```

Computes the log-probability of an action $\mathbf u$

$$
\log p(\mathbf u|\mathbf v, \mathbf u_{t-1}) = \big(\beta \mathbf v + \beta^\rho \mathbf u_{t-1}) - \log \sum_{v_i} e^{\beta \mathbf v_i + \beta^\rho u_{t-1}^{(i)}}
$$

Arguments:

- **x**: State vector of type `ndarray((nstates,))`

Returns:

Scalar log-probability

---




### StickySoftmaxPolicy.sample

```python
fitr.agents.policies.sample(self, x)
```

Samples from the action distribution

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

`ndarray((nstates,))` one-hot action vector

---



## EpsilonGreedyPolicy

```python
fitr.agents.policies.EpsilonGreedyPolicy()
```

A policy that takes the maximally valued action with probability $1-\epsilon$, otherwise chooses randomlyself.

Arguments:

- **epsilon**: Probability of not taking the action with highest value
- **rng**: `numpy.random.RandomState` object

---




### EpsilonGreedyPolicy.action_prob

```python
fitr.agents.policies.action_prob(self, x)
```

Creates vector of action probabilities for e-greedy policy

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

`ndarray((nstates,))` vector of action probabilities

---




### EpsilonGreedyPolicy.sample

```python
fitr.agents.policies.sample(self, x)
```

Samples from the action distribution

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

`ndarray((nstates,))` one-hot action vector

---



## ValueFunction

```python
fitr.agents.value_functions.ValueFunction()
```

A general value function object.

A value function here is task specific and consists of several attributes:

- `nstates`: The number of states in the task, $|\mathcal X|$
- `nactions`: Number of actions in the task, $|\mathcal U|$
- `V`: State value function $\mathbf v = \mathcal V(\mathbf x)$
- `Q`: State-action value function $\mathbf Q = \mathcal Q(\mathbf x, \mathbf u)$
- `etrace`: An eligibility trace (optional)

Note that in general we rely on matrix-vector notation for value functions, rather than function notation. Vectors in the mathematical typesetting are by default column vectors.

Arguments:

- **env**: A `fitr.environments.Graph`

---




### ValueFunction.Qmax

```python
fitr.agents.value_functions.Qmax(self, x)
```

Return maximal action value for given state

$$
\max_{u_i}\mathcal Q(\mathbf x, u_i) = \max_{\mathbf u'} \mathbf u'^\top \mathbf Q \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of the maximal action value at the given state

---




### ValueFunction.Qmean

```python
fitr.agents.value_functions.Qmean(self, x)
```

Return mean action value for given state

$$
Mean \big(\mathcal Q(\mathbf x, :)\big) = \frac{1}{|\mathcal U|} \mathbf 1^\top \mathbf Q \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of the maximal action value at the given state

---




### ValueFunction.Qx

```python
fitr.agents.value_functions.Qx(self, x)
```

Compute action values for a given state

$$
\mathcal Q(\mathbf x, :) = \mathbf Q \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

`ndarray((nactions,))` vector of values for actions in the given state

---




### ValueFunction.Vx

```python
fitr.agents.value_functions.Vx(self, x)
```

Compute value of state $\mathbf x$

$$
\mathcal V(\mathbf x) = \mathbf v^\top \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of state $\mathbf x$

---




### ValueFunction.uQx

```python
fitr.agents.value_functions.uQx(self, u, x)
```

Compute value of taking action $\mathbf u$ in state $\mathbf x$

$$
\mathcal Q(\mathbf x, \mathbf u) = \mathbf u^\top \mathbf Q \mathbf x
$$

Arguments:

- **u**: `ndarray((nactions,))` one-hot action vector
- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of action $\mathbf u$ in state $\mathbf x$

---




### ValueFunction.update

```python
fitr.agents.value_functions.update(self, x, u, r, x_, u_)
```

Updates the value function

In the context of the base `ValueFunction` class, this is merely a placeholder. The specific update rule will depend on the specific value function desired.

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector
- **u**: `ndarray((nactions,))` one-hot action vector
- **r**: Scalar reward
- **x_**: `ndarray((nstates,))` one-hot next-state vector
- **u_**: `ndarray((nactions,))` one-hot next-action vector

---



## DummyLearner

```python
fitr.agents.value_functions.DummyLearner()
```

A critic/value function for the random learner

This class actually contributes nothing except identifying that a value function has been chosen for an `Agent` object

Arguments:

- **env**: A `fitr.environments.Graph`

---




### DummyLearner.Qmax

```python
fitr.agents.value_functions.Qmax(self, x)
```

Return maximal action value for given state

$$
\max_{u_i}\mathcal Q(\mathbf x, u_i) = \max_{\mathbf u'} \mathbf u'^\top \mathbf Q \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of the maximal action value at the given state

---




### DummyLearner.Qmean

```python
fitr.agents.value_functions.Qmean(self, x)
```

Return mean action value for given state

$$
Mean \big(\mathcal Q(\mathbf x, :)\big) = \frac{1}{|\mathcal U|} \mathbf 1^\top \mathbf Q \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of the maximal action value at the given state

---




### DummyLearner.Qx

```python
fitr.agents.value_functions.Qx(self, x)
```

Compute action values for a given state

$$
\mathcal Q(\mathbf x, :) = \mathbf Q \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

`ndarray((nactions,))` vector of values for actions in the given state

---




### DummyLearner.Vx

```python
fitr.agents.value_functions.Vx(self, x)
```

Compute value of state $\mathbf x$

$$
\mathcal V(\mathbf x) = \mathbf v^\top \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of state $\mathbf x$

---




### DummyLearner.uQx

```python
fitr.agents.value_functions.uQx(self, u, x)
```

Compute value of taking action $\mathbf u$ in state $\mathbf x$

$$
\mathcal Q(\mathbf x, \mathbf u) = \mathbf u^\top \mathbf Q \mathbf x
$$

Arguments:

- **u**: `ndarray((nactions,))` one-hot action vector
- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of action $\mathbf u$ in state $\mathbf x$

---




### DummyLearner.update

```python
fitr.agents.value_functions.update(self, x, u, r, x_, u_)
```

Updates the value function

In the context of the base `ValueFunction` class, this is merely a placeholder. The specific update rule will depend on the specific value function desired.

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector
- **u**: `ndarray((nactions,))` one-hot action vector
- **r**: Scalar reward
- **x_**: `ndarray((nstates,))` one-hot next-state vector
- **u_**: `ndarray((nactions,))` one-hot next-action vector

---



## InstrumentalRescorlaWagnerLearner

```python
fitr.agents.value_functions.InstrumentalRescorlaWagnerLearner()
```

Learns an instrumental control policy through one-step error-driven updates of the state-action value function

The instrumental Rescorla-Wagner rule is as follows:

$$
\mathbf Q \gets \mathbf Q + \alpha \big(r - \mathbf u^\top \mathbf Q \mathbf x \big) \mathbf u \mathbf x^\top,
$$

where $0 < \alpha < 1$ is the learning rate, and where the reward prediction error (RPE) is $\delta = (r - \mathbf u^\top \mathbf Q \mathbf x)$.

$$

Arguments:

- **env**: A `fitr.environments.Graph`
- **learning_rate**: Learning rate $\alpha$

---




### InstrumentalRescorlaWagnerLearner.Qmax

```python
fitr.agents.value_functions.Qmax(self, x)
```

Return maximal action value for given state

$$
\max_{u_i}\mathcal Q(\mathbf x, u_i) = \max_{\mathbf u'} \mathbf u'^\top \mathbf Q \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of the maximal action value at the given state

---




### InstrumentalRescorlaWagnerLearner.Qmean

```python
fitr.agents.value_functions.Qmean(self, x)
```

Return mean action value for given state

$$
Mean \big(\mathcal Q(\mathbf x, :)\big) = \frac{1}{|\mathcal U|} \mathbf 1^\top \mathbf Q \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of the maximal action value at the given state

---




### InstrumentalRescorlaWagnerLearner.Qx

```python
fitr.agents.value_functions.Qx(self, x)
```

Compute action values for a given state

$$
\mathcal Q(\mathbf x, :) = \mathbf Q \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

`ndarray((nactions,))` vector of values for actions in the given state

---




### InstrumentalRescorlaWagnerLearner.Vx

```python
fitr.agents.value_functions.Vx(self, x)
```

Compute value of state $\mathbf x$

$$
\mathcal V(\mathbf x) = \mathbf v^\top \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of state $\mathbf x$

---




### InstrumentalRescorlaWagnerLearner.uQx

```python
fitr.agents.value_functions.uQx(self, u, x)
```

Compute value of taking action $\mathbf u$ in state $\mathbf x$

$$
\mathcal Q(\mathbf x, \mathbf u) = \mathbf u^\top \mathbf Q \mathbf x
$$

Arguments:

- **u**: `ndarray((nactions,))` one-hot action vector
- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of action $\mathbf u$ in state $\mathbf x$

---




### InstrumentalRescorlaWagnerLearner.update

```python
fitr.agents.value_functions.update(self, x, u, r, x_, u_)
```

Updates the value function

In the context of the base `ValueFunction` class, this is merely a placeholder. The specific update rule will depend on the specific value function desired.

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector
- **u**: `ndarray((nactions,))` one-hot action vector
- **r**: Scalar reward
- **x_**: `ndarray((nstates,))` one-hot next-state vector
- **u_**: `ndarray((nactions,))` one-hot next-action vector

---



## QLearner

```python
fitr.agents.value_functions.QLearner()
```

Learns an instrumental control policy through Q-learning

The Q-learning rule is as follows:

$$
\mathbf Q \gets \mathbf Q + \alpha \big(r + \gamma \max_{\mathbf u'} \mathbf u'^\top \mathbf Q \mathbf x' - \mathbf u^\top \mathbf Q \mathbf x \big) \mathbf z,
$$

where $0 < \alpha < 1$ is the learning rate, $0 \leq \gamma \leq 1$ is a discount factor, and where the reward prediction error (RPE) is $\delta = (r + \gamma \max_{\mathbf u'} \mathbf u'^\top \mathbf Q \mathbf x' - \mathbf u^\top \mathbf Q \mathbf x)$. We have also included an eligibility trace $\mathbf z$ defined as

$$
\mathbf z = \mathbf u \mathbf x^\top +  \gamma \lambda \mathbf z
$$

Arguments:

- **env**: A `fitr.environments.Graph`
- **learning_rate**: Learning rate $\alpha$
- **discount_factor**: Discount factor $\gamma$
- **trace_decay**: Eligibility trace decay $\lambda$

---




### QLearner.Qmax

```python
fitr.agents.value_functions.Qmax(self, x)
```

Return maximal action value for given state

$$
\max_{u_i}\mathcal Q(\mathbf x, u_i) = \max_{\mathbf u'} \mathbf u'^\top \mathbf Q \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of the maximal action value at the given state

---




### QLearner.Qmean

```python
fitr.agents.value_functions.Qmean(self, x)
```

Return mean action value for given state

$$
Mean \big(\mathcal Q(\mathbf x, :)\big) = \frac{1}{|\mathcal U|} \mathbf 1^\top \mathbf Q \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of the maximal action value at the given state

---




### QLearner.Qx

```python
fitr.agents.value_functions.Qx(self, x)
```

Compute action values for a given state

$$
\mathcal Q(\mathbf x, :) = \mathbf Q \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

`ndarray((nactions,))` vector of values for actions in the given state

---




### QLearner.Vx

```python
fitr.agents.value_functions.Vx(self, x)
```

Compute value of state $\mathbf x$

$$
\mathcal V(\mathbf x) = \mathbf v^\top \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of state $\mathbf x$

---




### QLearner.uQx

```python
fitr.agents.value_functions.uQx(self, u, x)
```

Compute value of taking action $\mathbf u$ in state $\mathbf x$

$$
\mathcal Q(\mathbf x, \mathbf u) = \mathbf u^\top \mathbf Q \mathbf x
$$

Arguments:

- **u**: `ndarray((nactions,))` one-hot action vector
- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of action $\mathbf u$ in state $\mathbf x$

---




### QLearner.update

```python
fitr.agents.value_functions.update(self, x, u, r, x_, u_)
```

Updates the value function

In the context of the base `ValueFunction` class, this is merely a placeholder. The specific update rule will depend on the specific value function desired.

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector
- **u**: `ndarray((nactions,))` one-hot action vector
- **r**: Scalar reward
- **x_**: `ndarray((nstates,))` one-hot next-state vector
- **u_**: `ndarray((nactions,))` one-hot next-action vector

---



## SARSALearner

```python
fitr.agents.value_functions.SARSALearner()
```

Learns an instrumental control policy through the SARSA learning rule

The SARSA learning rule is as follows:

$$
\mathbf Q \gets \mathbf Q + \alpha \big(r + \gamma \mathbf u'^\top \mathbf Q \mathbf x' - \mathbf u^\top \mathbf Q \mathbf x \big) \mathbf z,
$$

where $0 < \alpha < 1$ is the learning rate, $0 \leq \gamma \leq 1$ is a discount factor, and where the reward prediction error (RPE) is $\delta = (r + \gamma \mathbf u'^\top \mathbf Q \mathbf x' - \mathbf u^\top \mathbf Q \mathbf x)$. We have also included an eligibility trace $\mathbf z$ defined as

$$
\mathbf z = \mathbf u \mathbf x^\top +  \gamma \lambda \mathbf z
$$

Arguments:

- **env**: A `fitr.environments.Graph`
- **learning_rate**: Learning rate $\alpha$
- **discount_factor**: Discount factor $\gamma$
- **trace_decay**: Eligibility trace decay $\lambda$

---




### SARSALearner.Qmax

```python
fitr.agents.value_functions.Qmax(self, x)
```

Return maximal action value for given state

$$
\max_{u_i}\mathcal Q(\mathbf x, u_i) = \max_{\mathbf u'} \mathbf u'^\top \mathbf Q \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of the maximal action value at the given state

---




### SARSALearner.Qmean

```python
fitr.agents.value_functions.Qmean(self, x)
```

Return mean action value for given state

$$
Mean \big(\mathcal Q(\mathbf x, :)\big) = \frac{1}{|\mathcal U|} \mathbf 1^\top \mathbf Q \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of the maximal action value at the given state

---




### SARSALearner.Qx

```python
fitr.agents.value_functions.Qx(self, x)
```

Compute action values for a given state

$$
\mathcal Q(\mathbf x, :) = \mathbf Q \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

`ndarray((nactions,))` vector of values for actions in the given state

---




### SARSALearner.Vx

```python
fitr.agents.value_functions.Vx(self, x)
```

Compute value of state $\mathbf x$

$$
\mathcal V(\mathbf x) = \mathbf v^\top \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of state $\mathbf x$

---




### SARSALearner.uQx

```python
fitr.agents.value_functions.uQx(self, u, x)
```

Compute value of taking action $\mathbf u$ in state $\mathbf x$

$$
\mathcal Q(\mathbf x, \mathbf u) = \mathbf u^\top \mathbf Q \mathbf x
$$

Arguments:

- **u**: `ndarray((nactions,))` one-hot action vector
- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of action $\mathbf u$ in state $\mathbf x$

---




### SARSALearner.update

```python
fitr.agents.value_functions.update(self, x, u, r, x_, u_)
```

Updates the value function

In the context of the base `ValueFunction` class, this is merely a placeholder. The specific update rule will depend on the specific value function desired.

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector
- **u**: `ndarray((nactions,))` one-hot action vector
- **r**: Scalar reward
- **x_**: `ndarray((nstates,))` one-hot next-state vector
- **u_**: `ndarray((nactions,))` one-hot next-action vector

---



## Agent

```python
fitr.agents.agents.Agent()
```

Base class for synthetic RL agents.

Arguments:

meta : List of metadata of arbitrary type. e.g. labels, covariates, etc.
params : List of parameters for the agent. Should be filled for specific agent.

---




### Agent.action

```python
fitr.agents.agents.action(self, state)
```

Selects an action given the current state of environment.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector

---




### Agent.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```

Updates the model's parameters.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector
- **action**: `ndarray((nactions,))` one-hot action vector
- **reward**: scalar reward
- **next_state**: `ndarray((nstates,))` one-hot next-state vector
- **next_action**: `ndarray((nactions,))` one-hot action vector

---




### Agent.reset_trace

```python
fitr.agents.agents.reset_trace(self, x, u=None)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector
- **u**: `ndarray((nactions,))` one-hot action vector (optional)

---



## BanditAgent

```python
fitr.agents.agents.BanditAgent()
```

A base class for agents in bandit tasks (i.e. with one step).

Arguments:

- **task**: `fitr.environments.Graph`

---




### BanditAgent.action

```python
fitr.agents.agents.action(self, state)
```

Selects an action given the current state of environment.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector

---




### BanditAgent.generate_data

```python
fitr.agents.agents.generate_data(self, ntrials)
```

For the parent agent, this function generates data from a bandit task

Arguments:

- **ntrials**: `int` number of trials

Returns:

`fitr.data.BehaviouralData`

---




### BanditAgent.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```

Updates the model's parameters.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector
- **action**: `ndarray((nactions,))` one-hot action vector
- **reward**: scalar reward
- **next_state**: `ndarray((nstates,))` one-hot next-state vector
- **next_action**: `ndarray((nactions,))` one-hot action vector

---




### BanditAgent.log_prob

```python
fitr.agents.agents.log_prob(self, state)
```

Computes the log-likelihood over actions for a given state under the present agent parameters.

Presently this only works for the state-action value function. In all other cases, you should define your own log-likelihood function. However, this can be used as a template.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector

Returns:

`ndarray((nactions,))` log-likelihood vector

---




### BanditAgent.reset_trace

```python
fitr.agents.agents.reset_trace(self, x, u=None)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector
- **u**: `ndarray((nactions,))` one-hot action vector (optional)

---



## MDPAgent

```python
fitr.agents.agents.MDPAgent()
```

A base class for agents that operate on MDPs.

This mainly has implications for generating data.

Arguments:

- **task**: `fitr.environments.Graph`

---




### MDPAgent.action

```python
fitr.agents.agents.action(self, state)
```

Selects an action given the current state of environment.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector

---




### MDPAgent.generate_data

```python
fitr.agents.agents.generate_data(self, ntrials)
```

For the parent agent, this function generates data from a Markov Decision Process (MDP) task

Arguments:

- **ntrials**: `int` number of trials

Returns:

`fitr.data.BehaviouralData`

---




### MDPAgent.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```

Updates the model's parameters.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector
- **action**: `ndarray((nactions,))` one-hot action vector
- **reward**: scalar reward
- **next_state**: `ndarray((nstates,))` one-hot next-state vector
- **next_action**: `ndarray((nactions,))` one-hot action vector

---




### MDPAgent.reset_trace

```python
fitr.agents.agents.reset_trace(self, x, u=None)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector
- **u**: `ndarray((nactions,))` one-hot action vector (optional)

---



## RandomBanditAgent

```python
fitr.agents.agents.RandomBanditAgent()
```

An agent that simply selects random actions at each trial


---




### RandomBanditAgent.action

```python
fitr.agents.agents.action(self, state)
```

Selects an action given the current state of environment.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector

---




### RandomBanditAgent.generate_data

```python
fitr.agents.agents.generate_data(self, ntrials)
```

For the parent agent, this function generates data from a bandit task

Arguments:

- **ntrials**: `int` number of trials

Returns:

`fitr.data.BehaviouralData`

---




### RandomBanditAgent.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```

Updates the model's parameters.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector
- **action**: `ndarray((nactions,))` one-hot action vector
- **reward**: scalar reward
- **next_state**: `ndarray((nstates,))` one-hot next-state vector
- **next_action**: `ndarray((nactions,))` one-hot action vector

---




### RandomBanditAgent.log_prob

```python
fitr.agents.agents.log_prob(self, state)
```

Computes the log-likelihood over actions for a given state under the present agent parameters.

Presently this only works for the state-action value function. In all other cases, you should define your own log-likelihood function. However, this can be used as a template.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector

Returns:

`ndarray((nactions,))` log-likelihood vector

---




### RandomBanditAgent.reset_trace

```python
fitr.agents.agents.reset_trace(self, x, u=None)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector
- **u**: `ndarray((nactions,))` one-hot action vector (optional)

---



## RandomMDPAgent

```python
fitr.agents.agents.RandomMDPAgent()
```

An agent that simply selects random actions at each trial

Notes
-----
This has been specified as an `OnPolicyAgent` arbitrarily.

---




### RandomMDPAgent.action

```python
fitr.agents.agents.action(self, state)
```

Selects an action given the current state of environment.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector

---




### RandomMDPAgent.generate_data

```python
fitr.agents.agents.generate_data(self, ntrials)
```

For the parent agent, this function generates data from a Markov Decision Process (MDP) task

Arguments:

- **ntrials**: `int` number of trials

Returns:

`fitr.data.BehaviouralData`

---




### RandomMDPAgent.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```

Updates the model's parameters.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector
- **action**: `ndarray((nactions,))` one-hot action vector
- **reward**: scalar reward
- **next_state**: `ndarray((nstates,))` one-hot next-state vector
- **next_action**: `ndarray((nactions,))` one-hot action vector

---




### RandomMDPAgent.reset_trace

```python
fitr.agents.agents.reset_trace(self, x, u=None)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector
- **u**: `ndarray((nactions,))` one-hot action vector (optional)

---



## SARSASoftmaxAgent

```python
fitr.agents.agents.SARSASoftmaxAgent()
```

An agent that uses the SARSA learning rule and a softmax policy

The softmax policy selects actions from a multinomial

$$
\mathbf u \sim \mathrm{Multinomial}(1, \mathbf p=\varsigma(\mathbf v)),
$$

whose parameters are

$$
p(\mathbf u|\mathbf v) = \varsigma(\mathbf v) = \frac{e^{\beta \mathbf v}}{\sum_{i}^{|\mathbf v|} e^{\beta v_i}}.
$$

The value function is SARSA:

$$
\mathbf Q \gets \mathbf Q + \alpha \big(r + \gamma \mathbf u'^\top \mathbf Q \mathbf x' - \mathbf u^\top \mathbf Q \mathbf x \big) \mathbf z,
$$

where $0 < \alpha < 1$ is the learning rate, $0 \leq \gamma \leq 1$ is a discount factor, and where the reward prediction error (RPE) is $\delta = (r + \gamma \mathbf u'^\top \mathbf Q \mathbf x' - \mathbf u^\top \mathbf Q \mathbf x)$. We have also included an eligibility trace $\mathbf z$ defined as

$$
\mathbf z = \mathbf u \mathbf x^\top +  \gamma \lambda \mathbf z
$$

Arguments:

- **task**: `fitr.environments.Graph`
- **learning_rate**: Learning rate $\alpha$
- **discount_factor**: Discount factor $\gamma$
- **trace_decay**: Eligibility trace decay $\lambda$
- **inverse_softmax_temp**: Inverse softmax temperature $\beta$
- **rng**: `np.random.RandomState`

---




### SARSASoftmaxAgent.action

```python
fitr.agents.agents.action(self, state)
```

Selects an action given the current state of environment.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector

---




### SARSASoftmaxAgent.generate_data

```python
fitr.agents.agents.generate_data(self, ntrials)
```

For the parent agent, this function generates data from a Markov Decision Process (MDP) task

Arguments:

- **ntrials**: `int` number of trials

Returns:

`fitr.data.BehaviouralData`

---




### SARSASoftmaxAgent.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```

Updates the model's parameters.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector
- **action**: `ndarray((nactions,))` one-hot action vector
- **reward**: scalar reward
- **next_state**: `ndarray((nstates,))` one-hot next-state vector
- **next_action**: `ndarray((nactions,))` one-hot action vector

---




### SARSASoftmaxAgent.reset_trace

```python
fitr.agents.agents.reset_trace(self, x, u=None)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector
- **u**: `ndarray((nactions,))` one-hot action vector (optional)

---



## QLearningSoftmaxAgent

```python
fitr.agents.agents.QLearningSoftmaxAgent()
```

An agent that uses the Q-learning rule and a softmax policy

The softmax policy selects actions from a multinomial

$$
\mathbf u \sim \mathrm{Multinomial}(1, \mathbf p=\varsigma(\mathbf v)),
$$

whose parameters are

$$
p(\mathbf u|\mathbf v) = \varsigma(\mathbf v) = \frac{e^{\beta \mathbf v}}{\sum_{i}^{|\mathbf v|} e^{\beta v_i}}.
$$

The value function is Q-learning:

$$
\mathbf Q \gets \mathbf Q + \alpha \big(r + \gamma \max_{\mathbf u'} \mathbf u'^\top \mathbf Q \mathbf x' - \mathbf u^\top \mathbf Q \mathbf x \big) \mathbf z,
$$

where $0 < \alpha < 1$ is the learning rate, $0 \leq \gamma \leq 1$ is a discount factor, and where the reward prediction error (RPE) is $\delta = (r + \gamma \max_{\mathbf u'} \mathbf u'^\top \mathbf Q \mathbf x' - \mathbf u^\top \mathbf Q \mathbf x)$. The eligibility trace $\mathbf z$ is defined as

$$
\mathbf z = \mathbf u \mathbf x^\top +  \gamma \lambda \mathbf z
$$

Arguments:

- **task**: `fitr.environments.Graph`
- **learning_rate**: Learning rate $\alpha$
- **discount_factor**: Discount factor $\gamma$
- **trace_decay**: Eligibility trace decay $\lambda$
- **inverse_softmax_temp**: Inverse softmax temperature $\beta$
- **rng**: `np.random.RandomState`

---




### QLearningSoftmaxAgent.action

```python
fitr.agents.agents.action(self, state)
```

Selects an action given the current state of environment.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector

---




### QLearningSoftmaxAgent.generate_data

```python
fitr.agents.agents.generate_data(self, ntrials)
```

For the parent agent, this function generates data from a Markov Decision Process (MDP) task

Arguments:

- **ntrials**: `int` number of trials

Returns:

`fitr.data.BehaviouralData`

---




### QLearningSoftmaxAgent.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```

Updates the model's parameters.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector
- **action**: `ndarray((nactions,))` one-hot action vector
- **reward**: scalar reward
- **next_state**: `ndarray((nstates,))` one-hot next-state vector
- **next_action**: `ndarray((nactions,))` one-hot action vector

---




### QLearningSoftmaxAgent.reset_trace

```python
fitr.agents.agents.reset_trace(self, x, u=None)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector
- **u**: `ndarray((nactions,))` one-hot action vector (optional)

---



## RWSoftmaxAgent

```python
fitr.agents.agents.RWSoftmaxAgent()
```

An instrumental Rescorla-Wagner agent with a softmax policy

The softmax policy selects actions from a multinomial

$$
\mathbf u \sim \mathrm{Multinomial}(1, \mathbf p=\varsigma(\mathbf v)),
$$

whose parameters are

$$
p(\mathbf u|\mathbf v) = \varsigma(\mathbf v) = \frac{e^{\beta \mathbf v}}{\sum_{i}^{|\mathbf v|} e^{\beta v_i}}.
$$

The value function is the Rescorla-Wagner learning rule:

$$
\mathbf Q \gets \mathbf Q + \alpha \big(r - \mathbf u^\top \mathbf Q \mathbf x \big) \mathbf u \mathbf x^\top,
$$

where $0 < \alpha < 1$ is the learning rate, $0 \leq \gamma \leq 1$ is a discount factor, and where the reward prediction error (RPE) is $\delta = (r - \mathbf u^\top \mathbf Q \mathbf x)$.

Arguments:

- **task**: `fitr.environments.Graph`
- **learning_rate**: Learning rate $\alpha$
- **inverse_softmax_temp**: Inverse softmax temperature $\beta$
- **rng**: `np.random.RandomState`

---




### RWSoftmaxAgent.action

```python
fitr.agents.agents.action(self, state)
```

Selects an action given the current state of environment.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector

---




### RWSoftmaxAgent.generate_data

```python
fitr.agents.agents.generate_data(self, ntrials)
```

For the parent agent, this function generates data from a bandit task

Arguments:

- **ntrials**: `int` number of trials

Returns:

`fitr.data.BehaviouralData`

---




### RWSoftmaxAgent.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```

Updates the model's parameters.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector
- **action**: `ndarray((nactions,))` one-hot action vector
- **reward**: scalar reward
- **next_state**: `ndarray((nstates,))` one-hot next-state vector
- **next_action**: `ndarray((nactions,))` one-hot action vector

---




### RWSoftmaxAgent.log_prob

```python
fitr.agents.agents.log_prob(self, state)
```

Computes the log-likelihood over actions for a given state under the present agent parameters.

Presently this only works for the state-action value function. In all other cases, you should define your own log-likelihood function. However, this can be used as a template.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector

Returns:

`ndarray((nactions,))` log-likelihood vector

---




### RWSoftmaxAgent.reset_trace

```python
fitr.agents.agents.reset_trace(self, x, u=None)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector
- **u**: `ndarray((nactions,))` one-hot action vector (optional)

---



## RWSoftmaxAgentRewardSensitivity

```python
fitr.agents.agents.RWSoftmaxAgentRewardSensitivity()
```

An instrumental Rescorla-Wagner agent with a softmax policy, whose experienced reward is scaled by a factor $\rho$.

The softmax policy selects actions from a multinomial

$$
\mathbf u \sim \mathrm{Multinomial}(1, \mathbf p=\varsigma(\mathbf v)),
$$

whose parameters are

$$
p(\mathbf u|\mathbf v) = \varsigma(\mathbf v) = \frac{e^{\beta \mathbf v}}{\sum_{i}^{|\mathbf v|} e^{\beta v_i}}.
$$

The value function is the Rescorla-Wagner learning rule with scaled reward $\rho r$:

$$
\mathbf Q \gets \mathbf Q + \alpha \big(\rho r - \mathbf u^\top \mathbf Q \mathbf x \big) \mathbf u \mathbf x^\top,
$$

where $0 < \alpha < 1$ is the learning rate, $0 \leq \gamma \leq 1$ is a discount factor, and where the reward prediction error (RPE) is $\delta = (\rho r - \mathbf u^\top \mathbf Q \mathbf x)$.

Arguments:

- **task**: `fitr.environments.Graph`
- **learning_rate**: Learning rate $\alpha$
- **inverse_softmax_temp**: Inverse softmax temperature $\beta$
- **reward_sensitivity**: Reward sensitivity parameter $\rho$
- **rng**: `np.random.RandomState`

---




### RWSoftmaxAgentRewardSensitivity.action

```python
fitr.agents.agents.action(self, state)
```

Selects an action given the current state of environment.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector

---




### RWSoftmaxAgentRewardSensitivity.generate_data

```python
fitr.agents.agents.generate_data(self, ntrials)
```

For the parent agent, this function generates data from a bandit task

Arguments:

- **ntrials**: `int` number of trials

Returns:

`fitr.data.BehaviouralData`

---




### RWSoftmaxAgentRewardSensitivity.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```

Updates the model's parameters.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector
- **action**: `ndarray((nactions,))` one-hot action vector
- **reward**: scalar reward
- **next_state**: `ndarray((nstates,))` one-hot next-state vector
- **next_action**: `ndarray((nactions,))` one-hot action vector

---




### RWSoftmaxAgentRewardSensitivity.log_prob

```python
fitr.agents.agents.log_prob(self, state)
```

Computes the log-likelihood over actions for a given state under the present agent parameters.

Presently this only works for the state-action value function. In all other cases, you should define your own log-likelihood function. However, this can be used as a template.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector

Returns:

`ndarray((nactions,))` log-likelihood vector

---




### RWSoftmaxAgentRewardSensitivity.reset_trace

```python
fitr.agents.agents.reset_trace(self, x, u=None)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector
- **u**: `ndarray((nactions,))` one-hot action vector (optional)

---


