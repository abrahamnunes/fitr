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



---




### EpsilonGreedyPolicy.action_prob

```python
fitr.agents.policies.action_prob(self, x)
```

Creates vector of action probabilities for e-greedy policy 

---




### EpsilonGreedyPolicy.sample

```python
fitr.agents.policies.sample(self, x)
```



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



---




### ValueFunction.uQx

```python
fitr.agents.value_functions.uQx(self, u, x)
```



---



## DummyLearner

```python
fitr.agents.value_functions.DummyLearner()
```

A critic for the random learner 

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



---




### DummyLearner.uQx

```python
fitr.agents.value_functions.uQx(self, u, x)
```



---




### DummyLearner.update

```python
fitr.agents.value_functions.update(self, x, u, r, x_, u_)
```



---



## InstrumentalRescorlaWagnerLearner

```python
fitr.agents.value_functions.InstrumentalRescorlaWagnerLearner()
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



---




### InstrumentalRescorlaWagnerLearner.uQx

```python
fitr.agents.value_functions.uQx(self, u, x)
```



---




### InstrumentalRescorlaWagnerLearner.update

```python
fitr.agents.value_functions.update(self, x, u, r, x_, u_)
```



---



## QLearner

```python
fitr.agents.value_functions.QLearner()
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



---




### QLearner.uQx

```python
fitr.agents.value_functions.uQx(self, u, x)
```



---




### QLearner.update

```python
fitr.agents.value_functions.update(self, x, u, r, x_, u_)
```



---



## SARSALearner

```python
fitr.agents.value_functions.SARSALearner()
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



---




### SARSALearner.uQx

```python
fitr.agents.value_functions.uQx(self, u, x)
```



---




### SARSALearner.update

```python
fitr.agents.value_functions.update(self, x, u, r, x_, u_)
```



---



## Agent

```python
fitr.agents.agents.Agent()
```

Base class for synthetic RL agents

Arguments:

meta : List of metadata of arbitrary type. e.g. labels, covariates, etc.
params : List of parameters for the agent. Should be filled for specific agent.

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

This mainly has implications for generating data

---




### BanditAgent.generate_data

```python
fitr.agents.agents.generate_data(self, ntrials)
```



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

This mainly has implications for generating data

---




### MDPAgent.generate_data

```python
fitr.agents.agents.generate_data(self, ntrials)
```



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



---




### RandomBanditAgent.generate_data

```python
fitr.agents.agents.generate_data(self, ntrials)
```



---




### RandomBanditAgent.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```



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



---




### RandomMDPAgent.generate_data

```python
fitr.agents.agents.generate_data(self, ntrials)
```



---




### RandomMDPAgent.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```



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

---




### SARSASoftmaxAgent.action

```python
fitr.agents.agents.action(self, state)
```



---




### SARSASoftmaxAgent.generate_data

```python
fitr.agents.agents.generate_data(self, ntrials)
```



---




### SARSASoftmaxAgent.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```



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

---




### QLearningSoftmaxAgent.action

```python
fitr.agents.agents.action(self, state)
```



---




### QLearningSoftmaxAgent.generate_data

```python
fitr.agents.agents.generate_data(self, ntrials)
```



---




### QLearningSoftmaxAgent.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```



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

A base class for agents in bandit tasks (i.e. with one step).

This mainly has implications for generating data

---




### RWSoftmaxAgent.action

```python
fitr.agents.agents.action(self, state)
```



---




### RWSoftmaxAgent.generate_data

```python
fitr.agents.agents.generate_data(self, ntrials)
```



---




### RWSoftmaxAgent.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```



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

A base class for agents in bandit tasks (i.e. with one step).

This mainly has implications for generating data

---




### RWSoftmaxAgentRewardSensitivity.action

```python
fitr.agents.agents.action(self, state)
```



---




### RWSoftmaxAgentRewardSensitivity.generate_data

```python
fitr.agents.agents.generate_data(self, ntrials)
```



---




### RWSoftmaxAgentRewardSensitivity.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```



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


