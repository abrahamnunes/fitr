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

Computes the log-probability of an action $\mathbf u$, in addition to computing derivatives up to second order

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

- **x**: `ndarray((nactions,))` action value vector

Returns:

`ndarray((nactions,))` vector of action probabilities

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

- **x**: State vector of type `ndarray((nactions,))`

Returns:

Scalar log-probability

---




### StickySoftmaxPolicy.sample

```python
fitr.agents.policies.sample(self, x)
```

Samples from the action distribution

Arguments:

- **x**: `ndarray((nactions,))` action value vector

Returns:

`ndarray((nactions,))` one-hot action vector

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
- `rpe`: Reward prediction error history
- `etrace`: An eligibility trace (optional)
- `dV`: A dictionary storing gradients with respect to parameters (named keys)
- `dQ`: A dictionary storing gradients with respect to parameters (named keys)

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




### ValueFunction.grad_Qx

```python
fitr.agents.value_functions.grad_Qx(self, x)
```

Compute gradient of action values for a given state

$$
\mathcal Q(\mathbf x, :) = \mathbf Q \mathbf x,
$$

where the gradient is defined as

$$
\frac{\partial}{\partial \mathbf Q} \mathcal Q(\mathbf x, :) = \mathbf 1 \mathbf x^\top,
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

`ndarray((nactions,))` vector of values for actions in the given state

---




### ValueFunction.grad_Vx

```python
fitr.agents.value_functions.grad_Vx(self, x)
```

Compute the gradient of state value function with respect to parameters $\mathbf v$

$$
\mathcal V(\mathbf x) = \mathbf v^\top \mathbf x,
$$

where the gradient is defined as

$$
\nabla_{\mathbf v} \mathcal V(\mathbf x) = \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of state $\mathbf x$

---




### ValueFunction.grad_uQx

```python
fitr.agents.value_functions.grad_uQx(self, u, x)
```

Compute derivative of value of taking action $\mathbf u$ in state $\mathbf x$ with respect to value function parameters $\mathbf Q$

$$
\mathcal Q(\mathbf x, \mathbf u) = \mathbf u^\top \mathbf Q \mathbf x,
$$

where the derivative is defined as

$$
\frac{\partial}{\partial \mathbf Q} \mathcal Q(\mathbf x, \mathbf u) = \mathbf u \mathbf x^\top,
$$

Arguments:

- **u**: `ndarray((nactions,))` one-hot action vector
- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of action $\mathbf u$ in state $\mathbf x$

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




### DummyLearner.grad_Qx

```python
fitr.agents.value_functions.grad_Qx(self, x)
```

Compute gradient of action values for a given state

$$
\mathcal Q(\mathbf x, :) = \mathbf Q \mathbf x,
$$

where the gradient is defined as

$$
\frac{\partial}{\partial \mathbf Q} \mathcal Q(\mathbf x, :) = \mathbf 1 \mathbf x^\top,
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

`ndarray((nactions,))` vector of values for actions in the given state

---




### DummyLearner.grad_Vx

```python
fitr.agents.value_functions.grad_Vx(self, x)
```

Compute the gradient of state value function with respect to parameters $\mathbf v$

$$
\mathcal V(\mathbf x) = \mathbf v^\top \mathbf x,
$$

where the gradient is defined as

$$
\nabla_{\mathbf v} \mathcal V(\mathbf x) = \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of state $\mathbf x$

---




### DummyLearner.grad_uQx

```python
fitr.agents.value_functions.grad_uQx(self, u, x)
```

Compute derivative of value of taking action $\mathbf u$ in state $\mathbf x$ with respect to value function parameters $\mathbf Q$

$$
\mathcal Q(\mathbf x, \mathbf u) = \mathbf u^\top \mathbf Q \mathbf x,
$$

where the derivative is defined as

$$
\frac{\partial}{\partial \mathbf Q} \mathcal Q(\mathbf x, \mathbf u) = \mathbf u \mathbf x^\top,
$$

Arguments:

- **u**: `ndarray((nactions,))` one-hot action vector
- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of action $\mathbf u$ in state $\mathbf x$

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




### InstrumentalRescorlaWagnerLearner.grad_Qx

```python
fitr.agents.value_functions.grad_Qx(self, x)
```

Compute gradient of action values for a given state

$$
\mathcal Q(\mathbf x, :) = \mathbf Q \mathbf x,
$$

where the gradient is defined as

$$
\frac{\partial}{\partial \mathbf Q} \mathcal Q(\mathbf x, :) = \mathbf 1 \mathbf x^\top,
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

`ndarray((nactions,))` vector of values for actions in the given state

---




### InstrumentalRescorlaWagnerLearner.grad_Vx

```python
fitr.agents.value_functions.grad_Vx(self, x)
```

Compute the gradient of state value function with respect to parameters $\mathbf v$

$$
\mathcal V(\mathbf x) = \mathbf v^\top \mathbf x,
$$

where the gradient is defined as

$$
\nabla_{\mathbf v} \mathcal V(\mathbf x) = \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of state $\mathbf x$

---




### InstrumentalRescorlaWagnerLearner.grad_uQx

```python
fitr.agents.value_functions.grad_uQx(self, u, x)
```

Compute derivative of value of taking action $\mathbf u$ in state $\mathbf x$ with respect to value function parameters $\mathbf Q$

$$
\mathcal Q(\mathbf x, \mathbf u) = \mathbf u^\top \mathbf Q \mathbf x,
$$

where the derivative is defined as

$$
\frac{\partial}{\partial \mathbf Q} \mathcal Q(\mathbf x, \mathbf u) = \mathbf u \mathbf x^\top,
$$

Arguments:

- **u**: `ndarray((nactions,))` one-hot action vector
- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of action $\mathbf u$ in state $\mathbf x$

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

Computes the value function update of the instrumental Rescorla-Wagner learning rule and computes derivative with respect to the learning rate.

This derivative is defined as

$$
\frac{\partial}{\partial \alpha} \mathcal Q(\mathbf x, \mathbf u; \alpha) = \delta \mathbf u \mathbf x^\top + \frac{\partial}{\partial \alpha} \mathcal Q(\mathbf x, \mathbf u; \alpha) (1-\alpha \mathbf u \mathbf x^\top)
$$

and the second order derivative with respect to learning rate is

$$
\frac{\partial}{\partial \alpha} \mathcal Q(\mathbf x, \mathbf u; \alpha) = - 2 \mathbf u \mathbf x^\top \partial_\alpha \mathcal Q(\mathbf x, \mathbf u; \alpha) + \partial^2_\alpha \mathcal Q(\mathbf x, \mathbf u; \alpha) (1 - \alpha \mathbf u \mathbf x^\top)
$$

Arguments:

- **x**: `ndarray((nstates, ))`. State vector
- **u**: `ndarray((nactions, ))`. Action vector
- **r**: `float`. Reward received
- **x_**: `ndarray((nstates, ))`. For compatibility
- **u_**: `ndarray((nactions, ))`. For compatibility

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




### QLearner.grad_Qx

```python
fitr.agents.value_functions.grad_Qx(self, x)
```

Compute gradient of action values for a given state

$$
\mathcal Q(\mathbf x, :) = \mathbf Q \mathbf x,
$$

where the gradient is defined as

$$
\frac{\partial}{\partial \mathbf Q} \mathcal Q(\mathbf x, :) = \mathbf 1 \mathbf x^\top,
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

`ndarray((nactions,))` vector of values for actions in the given state

---




### QLearner.grad_Vx

```python
fitr.agents.value_functions.grad_Vx(self, x)
```

Compute the gradient of state value function with respect to parameters $\mathbf v$

$$
\mathcal V(\mathbf x) = \mathbf v^\top \mathbf x,
$$

where the gradient is defined as

$$
\nabla_{\mathbf v} \mathcal V(\mathbf x) = \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of state $\mathbf x$

---




### QLearner.grad_uQx

```python
fitr.agents.value_functions.grad_uQx(self, u, x)
```

Compute derivative of value of taking action $\mathbf u$ in state $\mathbf x$ with respect to value function parameters $\mathbf Q$

$$
\mathcal Q(\mathbf x, \mathbf u) = \mathbf u^\top \mathbf Q \mathbf x,
$$

where the derivative is defined as

$$
\frac{\partial}{\partial \mathbf Q} \mathcal Q(\mathbf x, \mathbf u) = \mathbf u \mathbf x^\top,
$$

Arguments:

- **u**: `ndarray((nactions,))` one-hot action vector
- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of action $\mathbf u$ in state $\mathbf x$

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

Computes value function updates and their derivatives for the Q-learning model 

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




### SARSALearner.grad_Qx

```python
fitr.agents.value_functions.grad_Qx(self, x)
```

Compute gradient of action values for a given state

$$
\mathcal Q(\mathbf x, :) = \mathbf Q \mathbf x,
$$

where the gradient is defined as

$$
\frac{\partial}{\partial \mathbf Q} \mathcal Q(\mathbf x, :) = \mathbf 1 \mathbf x^\top,
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

`ndarray((nactions,))` vector of values for actions in the given state

---




### SARSALearner.grad_Vx

```python
fitr.agents.value_functions.grad_Vx(self, x)
```

Compute the gradient of state value function with respect to parameters $\mathbf v$

$$
\mathcal V(\mathbf x) = \mathbf v^\top \mathbf x,
$$

where the gradient is defined as

$$
\nabla_{\mathbf v} \mathcal V(\mathbf x) = \mathbf x
$$

Arguments:

- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of state $\mathbf x$

---




### SARSALearner.grad_uQx

```python
fitr.agents.value_functions.grad_uQx(self, u, x)
```

Compute derivative of value of taking action $\mathbf u$ in state $\mathbf x$ with respect to value function parameters $\mathbf Q$

$$
\mathcal Q(\mathbf x, \mathbf u) = \mathbf u^\top \mathbf Q \mathbf x,
$$

where the derivative is defined as

$$
\frac{\partial}{\partial \mathbf Q} \mathcal Q(\mathbf x, \mathbf u) = \mathbf u \mathbf x^\top,
$$

Arguments:

- **u**: `ndarray((nactions,))` one-hot action vector
- **x**: `ndarray((nstates,))` one-hot state vector

Returns:

Scalar value of action $\mathbf u$ in state $\mathbf x$

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

Computes value function updates and their derivatives for the SARSA model 

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

Updates the model's parameters and computes gradients

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
fitr.agents.agents.reset_trace(self, state_only=False)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **state_only**: `bool`. If the eligibility trace is only an `nstate` dimensional vector (i.e. for a Pavlovian conditioning model) then set to `True`. For instumental models, the eligibility trace should be an `nactions` by `nstates` matrix, so keep this to `False` in that case.

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

Updates the model's parameters and computes gradients

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
fitr.agents.agents.reset_trace(self, state_only=False)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **state_only**: `bool`. If the eligibility trace is only an `nstate` dimensional vector (i.e. for a Pavlovian conditioning model) then set to `True`. For instumental models, the eligibility trace should be an `nactions` by `nstates` matrix, so keep this to `False` in that case.

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
fitr.agents.agents.generate_data(self, ntrials, state_only=False)
```

For the parent agent, this function generates data from a Markov Decision Process (MDP) task

Arguments:

- **ntrials**: `int` number of trials
- **state_only**: `bool`. If the eligibility trace is only an `nstate` dimensional vector (i.e. for a Pavlovian conditioning model) then set to `True`. For instumental models, the eligibility trace should be an `nactions` by `nstates` matrix, so keep this to `False` in that case.

Returns:

`fitr.data.BehaviouralData`

---




### MDPAgent.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```

Updates the model's parameters and computes gradients

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
fitr.agents.agents.reset_trace(self, state_only=False)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **state_only**: `bool`. If the eligibility trace is only an `nstate` dimensional vector (i.e. for a Pavlovian conditioning model) then set to `True`. For instumental models, the eligibility trace should be an `nactions` by `nstates` matrix, so keep this to `False` in that case.

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

Updates the model's parameters and computes gradients

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
fitr.agents.agents.reset_trace(self, state_only=False)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **state_only**: `bool`. If the eligibility trace is only an `nstate` dimensional vector (i.e. for a Pavlovian conditioning model) then set to `True`. For instumental models, the eligibility trace should be an `nactions` by `nstates` matrix, so keep this to `False` in that case.

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
fitr.agents.agents.generate_data(self, ntrials, state_only=False)
```

For the parent agent, this function generates data from a Markov Decision Process (MDP) task

Arguments:

- **ntrials**: `int` number of trials
- **state_only**: `bool`. If the eligibility trace is only an `nstate` dimensional vector (i.e. for a Pavlovian conditioning model) then set to `True`. For instumental models, the eligibility trace should be an `nactions` by `nstates` matrix, so keep this to `False` in that case.

Returns:

`fitr.data.BehaviouralData`

---




### RandomMDPAgent.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```

Updates the model's parameters and computes gradients

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
fitr.agents.agents.reset_trace(self, state_only=False)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **state_only**: `bool`. If the eligibility trace is only an `nstate` dimensional vector (i.e. for a Pavlovian conditioning model) then set to `True`. For instumental models, the eligibility trace should be an `nactions` by `nstates` matrix, so keep this to `False` in that case.

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
fitr.agents.agents.generate_data(self, ntrials, state_only=False)
```

For the parent agent, this function generates data from a Markov Decision Process (MDP) task

Arguments:

- **ntrials**: `int` number of trials
- **state_only**: `bool`. If the eligibility trace is only an `nstate` dimensional vector (i.e. for a Pavlovian conditioning model) then set to `True`. For instumental models, the eligibility trace should be an `nactions` by `nstates` matrix, so keep this to `False` in that case.

Returns:

`fitr.data.BehaviouralData`

---




### SARSASoftmaxAgent.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```

Updates the model's parameters and computes gradients

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector
- **action**: `ndarray((nactions,))` one-hot action vector
- **reward**: scalar reward
- **next_state**: `ndarray((nstates,))` one-hot next-state vector
- **next_action**: `ndarray((nactions,))` one-hot action vector

---




### SARSASoftmaxAgent.log_prob

```python
fitr.agents.agents.log_prob(self, state, action)
```

Computes the log-probability of the given action and state under the model, while also computing first and second order derivatives.

This model has four free parameters:

- Learning rate $\alpha$
- Inverse softmax temperature $\beta$
- Discount factor $\gamma$
- Trace decay $\lambda$

__First-order partial derivatives__

We can break down the computation using the chain rule to reuse previously computed derivatives:

$$
\pd{\cL}{\alpha}  = \pd{\cL}{\logits} \pd{\logits}{\mathbf q} \pd{\mathbf q}{\mathbf Q} \pd{\mathbf Q}{\alpha}
$$

$$
\pd{\cL}{\beta}   = \pd{\cL}{\logits} \pd{\logits}{\beta}
$$

$$
\pd{\cL}{\gamma}  = \pd{\cL}{\logits} \pd{\logits}{\mathbf q} \pd{\mathbf q}{\mathbf Q} \pd{\mathbf Q}{\gamma}
$$

$$
\pd{\cL}{\lambda} = \pd{\cL}{\logits} \pd{\logits}{\mathbf q} \pd{\mathbf q}{\mathbf Q} \pd{\mathbf Q}{\lambda}
$$

_Action Probabilities_

$$
\partial_\alpha \varsigma = \pd{\varsigma}{\logits} \pd{\logits}{\mathbf q} \pd{\mathbf q}{\mathbf Q} \big( \partial_\alpha \mathbf Q \big) = \beta \big(\partial_{\logits} \varsigma \big)_i \big( \partial_\alpha Q \big)^i_j x^j
$$

_Value Function_

$$
\partial_\alpha Q_{ij} =  \partial_\alpha Q_{ij} + (\delta + \alpha \partial_\alpha \delta) z_{ij}
$$

$$
\partial_\gamma Q_{ij} =  \partial_\gamma Q_{ij} + \alpha \big( (\partial_\gamma \delta) z_{ij} + \delta (\partial_\gamma z_{ij}) \big)
$$

$$
\partial_\lambda Q_{ij} =  \partial_\lambda Q_{ij} + \alpha \big( (\partial_\lambda \delta) z_{ij} + \delta (\partial_\lambda z_{ij}) \big)
$$

_Reward Prediction Error_

$$
\partial_\alpha \delta = (\partial_{\mathbf Q} \delta)_{ij} (\partial_\alpha Q)^{ij}
$$

$$
\partial_\gamma \delta = (\partial_{\mathbf Q} \delta)_{ij} (\partial_\gamma Q)^{ij} + \tilde{u}_i Q^i_j \tilde{x}^j
$$

$$
\partial_\lambda \delta = (\partial_{\mathbf Q} \delta)_{ij} (\partial_\lambda Q)^{ij}
$$

_Trace Decay_

$$
\partial_\gamma z_{ij} = \lambda \big(z_{ij} + \gamma (\partial_\gamma z_{ij}) \big)
$$

$$
\partial_\lambda z_{ij} = \gamma \big(z_{ij} + \lambda (\partial_\lambda z_{ij}) \big)
$$

_Simplified Components of the Gradient Vector_

$$
\pd{\cL}{\alpha}  = \beta \big[\mathbf u - \varsigma(\logits) \big]_i \big( \partial_\alpha Q \big)^i_j x^j   = \beta \big[ u_i (\partial_\alpha Q)^i_j x^j - p(u_i) (\partial_\alpha Q)^i_j x^j \big]
$$

$$
\pd{\cL}{\beta}   =  \big[\mathbf u - \varsigma(\logits)\big]_i Q^i_j x^j = u_i Q^i_j x^j - p(u_i) Q^i_j x^j
$$

$$
\pd{\cL}{\gamma}  = \beta \big[\mathbf u - \varsigma(\logits) \big]_i \big( \partial_\gamma Q \big)^i_j x^j
$$

$$
\pd{\cL}{\lambda} = \beta \big[\mathbf u - \varsigma(\logits) \big]_i \big( \partial_\lambda Q \big)^i_j x^j
$$

__Second-Order Partial Derivatives__

The Hessian matrix for this model is

$$
\mathbf H = \left[
\begin{array}{cccc}
\pHd{\cL}{\alpha}                  & \pHo{\cL}{\alpha}{\beta}  & \pHo{\cL}{\alpha}{\gamma}  & \pHo{\cL}{\alpha}{\lambda} \\
\pHo{\cL}{\beta}{\alpha}   & \pHd{\cL}{\beta}              & \pHo{\cL}{\beta}{\gamma}   & \pHo{\cL}{\beta}{\lambda}  \\
\pHo{\cL}{\gamma}{\alpha}  & \pHo{\cL}{\gamma}{\beta}  & \pHd{\cL}{\gamma}                  & \pHo{\cL}{\gamma}{\lambda} \\
\pHo{\cL}{\lambda}{\alpha} & \pHo{\cL}{\lambda}{\beta} & \pHo{\cL}{\lambda}{\gamma} & \pHd{\cL}{\lambda}                 \\
\end{array}\right],
$$

where the second-order partial derivatives are such that $\mathbf H$ is symmetrical. We must therefore compute 10 second order partial derivatives, shown below:

$$
\pHd{\cL}{\alpha} = \beta \Big[ (\mathbf u - \varsigma(\logits))_i \big( \partial^2_\alpha Q \big)^i - \big( \partial_\alpha \varsigma \big)_j \big( \partial_\alpha Q)^j_k x^k \Big]_l x^l
$$

$$
\pHd{\cL}{\beta} = \Big( q_i \varsigma(\logits)^i \Big)^2 - \mathbf q \odot \mathbf q \odot \varsigma(\logits)
$$

$$
\pHd{\cL}{\gamma}  = \beta \Big[ (\mathbf u - \varsigma(\logits))_i \big( \partial^2_\gamma Q \big)^i - \big( \partial_\gamma \varsigma \big)_j \big( \partial_\gamma Q)^j_k x^k \Big]_l x^l
$$

$$
\pHd{\cL}{\lambda}  = \beta \Big[ (\mathbf u - \varsigma(\logits))_i \big( \partial^2_\lambda Q \big)^i - \big( \partial_\lambda \varsigma \big)_j \big( \partial_\lambda Q)^j_k x^k \Big]_l x^l
$$

The off diagonal elements of the Hessian are as follows:

$$
\pHo{\cL}{\alpha}{\beta}   = \bigg(\mathbf u - \varsigma(\logits) - \beta \big(\partial_\beta \varsigma \big) \bigg)_i \big( \partial_\alpha Q \big)^i_j x^j
$$

$$
\pHo{\cL}{\beta}{\gamma}   =  \bigg(\mathbf u - \varsigma(\logits) - \beta \big(\partial_\beta \varsigma \big) \bigg)_i \big( \partial_\gamma Q \big)^i_j x^j
$$

$$
\pHo{\cL}{\beta}{\lambda}  =  \bigg(\mathbf u - \varsigma(\logits) - \beta \big(\partial_\beta \varsigma \big) \bigg)_i \big( \partial_\lambda Q \big)^i_j x^j
$$

$$
\pHo{\cL}{\alpha}{\gamma}  = \beta \Big((\mathbf u - \varsigma(\logits))_i \big(\partial_\alpha \partial_\gamma Q \big)^i - \big( \partial_\gamma \varsigma \big)_j \big( \partial_\alpha Q \big)^j \Big)_k x^k
$$

$$
\pHo{\cL}{\alpha}{\lambda} =  \beta \Big((\mathbf u - \varsigma(\logits))_i \big(\partial_\alpha \partial_\lambda Q \big)^i - \big( \partial_\lambda \varsigma \big)_j \big( \partial_\alpha Q \big)^j \Big)_k x^k
$$

$$
\pHo{\cL}{\gamma}{\lambda} =  \beta \Big((\mathbf u - \varsigma(\logits))_i \big(\partial_\lambda \partial_\gamma Q \big)^i - \big( \partial_\lambda \varsigma \big)_j \big( \partial_\gamma Q \big)^j \Big)_k x^k
$$

_Reward Prediction Error_

$$
\partial^2_\alpha \delta = (\partial_{\mathbf Q} \delta)_{ij} (\partial^2_\alpha Q)^{ij}
$$

$$
\partial^2_\gamma \delta = (\partial_{\mathbf Q} \delta)_{ij} (\partial^2_\gamma Q)^{ij} + 2 \tilde{u}_i \big(\partial_\gamma Q\big)^i_j \tilde{x}^j
$$

$$
\partial^2_\lambda \delta = (\partial_{\mathbf Q} \delta)_{ij} (\partial^2_\lambda Q)^{ij}
$$

$$
\partial_\alpha \partial_\gamma \delta = (\partial_{\mathbf Q} \delta)_{ij} (\partial_\gamma \partial_\alpha Q)^{ij} + \tilde{u}_i \big(\partial_\alpha Q\big)^i_j \tilde{x}^j
$$

$$
\partial_\alpha \partial_\lambda \delta = (\partial_{\mathbf Q} \delta)_{ij} (\partial_\alpha \partial_\lambda Q)^{ij}
$$

$$
\partial_\gamma \partial_\lambda \delta = (\partial_{\mathbf Q} \delta)_{ij} (\partial_\gamma \partial_\lambda Q)^{ij} + \tilde{u}_i \big(\partial_\lambda Q\big)^i_j \tilde{x}^j
$$

_Value Function_

$$
\partial^2_\alpha Q_{ij} = \partial^2_\alpha Q_{ij} + 2(\partial_\alpha \delta) z_{ij} + \alpha (\partial^2_\alpha \delta) z_{ij}
$$

$$
\partial^2_\gamma Q_{ij} = \partial^2_\gamma Q_{ij} +  \alpha \Big( \big(\partial^2_\gamma \delta \big)z_{ij} +  \big(\partial_\gamma \delta \big) \big(\partial_\gamma z_{ij}\big) + \big(\partial_\gamma \delta \big) \big(\partial^2_\gamma z_{ij}\big) \Big)
$$

$$
\partial^2_\lambda Q_{ij} = \partial^2_\lambda Q_{ij} +  \alpha \Big( \big(\partial^2_\lambda \delta \big)z_{ij} +  \big(\partial_\lambda \delta \big) \big(\partial_\lambda z_{ij}\big) + \big(\partial_\lambda \delta \big) \big(\partial^2_\lambda z_{ij}\big) \Big)
$$

$$
\partial_\alpha \partial_\gamma Q_{ij} =  \partial_\alpha \partial_\gamma Q_{ij} + (\partial_\gamma \delta) z_{ij} + \delta \big(\partial_\gamma z_{ij} \big) + \alpha(\partial_\alpha \delta)\big(\partial_\gamma z_{ij} \big) + \alpha(\partial_\alpha \partial_\gamma \delta) z_{ij}
$$

$$
\partial_\alpha \partial_\lambda Q_{ij} = \partial_\alpha \partial_\lambda Q_{ij} + (\partial_\lambda \delta) z_{ij} + \delta \big(\partial_\lambda z_{ij} \big) + \alpha(\partial_\alpha \delta)\big(\partial_\lambda z_{ij} \big) + \alpha(\partial_\alpha \partial_\lambda \delta) z_{ij}
$$

$$
\partial_\gamma \partial_\lambda Q_{ij} = \partial_\gamma \partial_\lambda Q_{ij} + \alpha \Big[ \big( \partial_\lambda \partial_\gamma \delta \big) z_{ij} + \big( \partial_\gamma \delta \big)\big(\partial_\lambda z_{ij} \big) + \big(\partial_\lambda \delta \big)\big(\partial_\gamma z_{ij} \big) + \delta \big(\partial_\lambda \partial_\gamma z_{ij} \big) \Big]
$$

_Trace Decay_

$$
\partial^2_\gamma z = \lambda \Big( 2\big(\partial_\gamma z\big) + \gamma \big(\partial^2_\gamma z \big) \Big)
$$

$$
\partial^2_\lambda z = \gamma \Big( 2\big(\partial_\lambda z\big) + \lambda \big(\partial^2_\lambda z \big) \Big)
$$

$$
\partial_\gamma \partial_\lambda z = z  + \gamma \big( \partial_\gamma z \big) + \lambda \big( \partial_\lambda z \big) + \lambda \gamma \big( \partial_\gamma \partial_\lambda z \big)
$$

Arguments:

- **action**: `ndarray(nactions)`. One-hot action vector
- **state**: `ndarray(nstates)`. One-hot state vector

---




### SARSASoftmaxAgent.reset_trace

```python
fitr.agents.agents.reset_trace(self, state_only=False)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **state_only**: `bool`. If the eligibility trace is only an `nstate` dimensional vector (i.e. for a Pavlovian conditioning model) then set to `True`. For instumental models, the eligibility trace should be an `nactions` by `nstates` matrix, so keep this to `False` in that case.

---



## SARSAStickySoftmaxAgent

```python
fitr.agents.agents.SARSAStickySoftmaxAgent()
```

An agent that uses the SARSA learning rule and a sticky softmax policy

The sticky softmax policy selects actions from a multinomial

$$
\mathbf u \sim \mathrm{Multinomial}(1, \mathbf p=\varsigma(\mathbf v)),
$$

whose parameters are

$$
p(\mathbf u|\mathbf v, \mathbf u_{t-1}) = \varsigma(\mathbf v, \mathbf u_{t-1}) = \frac{e^{\beta \mathbf v + \beta^\rho \mathbf u_{t-1}}}{\sum_{i}^{|\mathbf v|} e^{\beta v_i + \beta^\rho u_{t-1}^{(i)}}}.
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
- **perseveration**: Perseveration parameter $\beta^\rho$
- **rng**: `np.random.RandomState`

---




### SARSAStickySoftmaxAgent.action

```python
fitr.agents.agents.action(self, state)
```

Selects an action given the current state of environment.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector

---




### SARSAStickySoftmaxAgent.generate_data

```python
fitr.agents.agents.generate_data(self, ntrials, state_only=False)
```

For the parent agent, this function generates data from a Markov Decision Process (MDP) task

Arguments:

- **ntrials**: `int` number of trials
- **state_only**: `bool`. If the eligibility trace is only an `nstate` dimensional vector (i.e. for a Pavlovian conditioning model) then set to `True`. For instumental models, the eligibility trace should be an `nactions` by `nstates` matrix, so keep this to `False` in that case.

Returns:

`fitr.data.BehaviouralData`

---




### SARSAStickySoftmaxAgent.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```

Updates the model's parameters and computes gradients

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector
- **action**: `ndarray((nactions,))` one-hot action vector
- **reward**: scalar reward
- **next_state**: `ndarray((nstates,))` one-hot next-state vector
- **next_action**: `ndarray((nactions,))` one-hot action vector

---




### SARSAStickySoftmaxAgent.reset_trace

```python
fitr.agents.agents.reset_trace(self, state_only=False)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **state_only**: `bool`. If the eligibility trace is only an `nstate` dimensional vector (i.e. for a Pavlovian conditioning model) then set to `True`. For instumental models, the eligibility trace should be an `nactions` by `nstates` matrix, so keep this to `False` in that case.

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
fitr.agents.agents.generate_data(self, ntrials, state_only=False)
```

For the parent agent, this function generates data from a Markov Decision Process (MDP) task

Arguments:

- **ntrials**: `int` number of trials
- **state_only**: `bool`. If the eligibility trace is only an `nstate` dimensional vector (i.e. for a Pavlovian conditioning model) then set to `True`. For instumental models, the eligibility trace should be an `nactions` by `nstates` matrix, so keep this to `False` in that case.

Returns:

`fitr.data.BehaviouralData`

---




### QLearningSoftmaxAgent.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```

Updates the model's parameters and computes gradients

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
fitr.agents.agents.reset_trace(self, state_only=False)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **state_only**: `bool`. If the eligibility trace is only an `nstate` dimensional vector (i.e. for a Pavlovian conditioning model) then set to `True`. For instumental models, the eligibility trace should be an `nactions` by `nstates` matrix, so keep this to `False` in that case.

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

Updates the model's parameters and computes gradients

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
fitr.agents.agents.log_prob(self, state, action)
```

Computes the log-probability of an action taken by the agent in a given state, as well as updates all partial derivatives with respect to the parameters.

This function overrides the `log_prob` method of the parent class.

Let

- $n_u \in \mathbb N_+$ be the dimensionality of the action space
- $n_x \in \mathbb N_+$ be the dimensionality of the state space
- $\mathbf u = (u_0, u_1, u_{n_u})^\top$ be a one-hot action vector
- $\mathbf x = (x_0, x_1, x_{n_x})^\top$ be a one-hot action vector
- $\mathbf Q \in \mathbb R^{n_u \times n_x}$ be the state-action value function parameters
- $\beta \in \mathbb R$ be the inverse softmax temperature
- $\alpha \in [0, 1]$ be the learning rate
- $\varsigma(\boldsymbol\pi) = p(\mathbf u | \mathbf Q, \beta)$ be a softmax function with logits $\pi_i = \beta Q_{ij} x^j$ (shown in Einstein summation convention).
- $\mathcal L = \log p(\mathbf u | \mathbf Q, \beta)$ be the log-likelihood function for trial $t$
- $q_i = Q_{ij} x^j$ be the value of the state $x^j$
- $v^i = e^{\beta q_i}$ be the softmax potential
- $\eta(\boldsymbol\pi)$ be the softmax partition function.

Then we have the partial derivative of $\mathcal L$ at trial $t$ with respect to $\alpha$

$$
\partial_{\alpha} \mathcal L = \beta \Big[ \big(\mathbf u - \varsigma(\pi)\big)_i (\partial_{\alpha} Q)^i_j x^j \Big],
$$

and with respect to $\beta$

$$
\partial_{\beta} \mathcal L = u_i \Big(\mathbf I_{n_u \times n_u} - \varsigma(\boldsymbol\pi)\Big)^i_j Q_{jk} x^k.
$$

We also compute the Hessian $\mathbf H$, defined as

$$
\mathbf H = \left[
\begin{array}{cc}
\partial^2_{\alpha} \mathcal L & \partial_{\alpha} \partial_{\beta} \mathcal L \\
\partial_{\beta} \partial_{\alpha} \mathcal L & \partial^2_{\beta} \mathcal L \\
\end{array}\right].
$$

The components of $\mathbf H$ are

$$
\partial^2_{\alpha} \mathcal L = \beta \Big( (\mathbf u - \varsigma(\boldsymbol\pi))_i (\partial^2_\alpha \mathbf Q)^i - \partial_{\alpha} \varsigma(\boldsymbol\pi)_i (\partial_{\alpha} \mathbf Q)^i \Big)_j x^j,
$$

$$
\partial^2_{\beta} \mathcal L = u_i \Big( \Big),
$$

$$
\partial_{\alpha} \partial_{\beta} \mathcal L = \Bigg[ (u - \varsigma(\boldsymbol\pi)) - \beta \partial_{\beta} \varsigma(\boldsymbol\pi) \Bigg]_i (\partial_{\alpha} Q)^i_k x^k.
$$

and where $\partial_{\beta} \partial_{\alpha} \mathcal L = \partial_{\alpha} \partial_{\beta} \mathcal L$ since the second derivatives of $\mathcal L$ are continuous in the neighbourhood of the parameters.

Arguments:

- **action**: `ndarray(nactions)`. One-hot action vector
- **state**: `ndarray(nstates)`. One-hot state vector

---




### RWSoftmaxAgent.reset_trace

```python
fitr.agents.agents.reset_trace(self, state_only=False)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **state_only**: `bool`. If the eligibility trace is only an `nstate` dimensional vector (i.e. for a Pavlovian conditioning model) then set to `True`. For instumental models, the eligibility trace should be an `nactions` by `nstates` matrix, so keep this to `False` in that case.

---



## RWStickySoftmaxAgent

```python
fitr.agents.agents.RWStickySoftmaxAgent()
```

An instrumental Rescorla-Wagner agent with a 'sticky' softmax policy

The softmax policy selects actions from a multinomial

$$
\mathbf u \sim \mathrm{Multinomial}(1, \mathbf p=\varsigma(\mathbf v, \mathbf u_{t-1})).
$$

whose parameters are

$$
p(\mathbf u|\mathbf v, \mathbf u_{t-1}) = \varsigma(\mathbf v, \mathbf u_{t-1}) = \frac{e^{\beta \mathbf v + \beta^\rho \mathbf u_{t-1}}}{\sum_{i}^{|\mathbf v|} e^{\beta v_i + \beta^\rho u_{t-1}^{(i)}}}.
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
- **perseveration**: Perseveration parameter $\beta^ho$
- **rng**: `np.random.RandomState`

---




### RWStickySoftmaxAgent.action

```python
fitr.agents.agents.action(self, state)
```

Selects an action given the current state of environment.

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector

---




### RWStickySoftmaxAgent.generate_data

```python
fitr.agents.agents.generate_data(self, ntrials)
```

For the parent agent, this function generates data from a bandit task

Arguments:

- **ntrials**: `int` number of trials

Returns:

`fitr.data.BehaviouralData`

---




### RWStickySoftmaxAgent.learning

```python
fitr.agents.agents.learning(self, state, action, reward, next_state, next_action)
```

Updates the model's parameters and computes gradients

The implementation will vary depending on the type of agent and environment.

Arguments:

- **state**: `ndarray((nstates,))` one-hot state vector
- **action**: `ndarray((nactions,))` one-hot action vector
- **reward**: scalar reward
- **next_state**: `ndarray((nstates,))` one-hot next-state vector
- **next_action**: `ndarray((nactions,))` one-hot action vector

---




### RWStickySoftmaxAgent.log_prob

```python
fitr.agents.agents.log_prob(self, state, action)
```

Computes the log-probability of an action taken by the agent in a given state, as well as updates all partial derivatives with respect to the parameters.

This function overrides the `log_prob` method of the parent class.

Let

- $n_u \in \mathbb N_+$ be the dimensionality of the action space
- $n_x \in \mathbb N_+$ be the dimensionality of the state space
- $\mathbf u = (u_0, u_1, u_{n_u})^\top$ be a one-hot action vector
- $\tilde{\mathbf u}$ be a one-hot vector representing the last trial's action, where at trial 0, $\tilde{\mathbf u} = \mathbf 0$.
- $\mathbf x = (x_0, x_1, x_{n_x})^\top$ be a one-hot action vector
- $\mathbf Q \in \mathbb R^{n_u \times n_x}$ be the state-action value function parameters
- $\beta \in \mathbb R$ be the inverse softmax temperature scaling the action values
- $\rho \in \mathbb R$ be the inverse softmax temperature scaling the influence of the past trial's action
- $\alpha \in [0, 1]$ be the learning rate
- $\varsigma(\boldsymbol\pi) = p(\mathbf u | \mathbf Q, \beta, \rho)$ be a softmax function with logits $\pi_i = \beta Q_{ij} x^j + \rho \tilde{u}_i$ (shown in Einstein summation convention).
- $\mathcal L = \log p(\mathbf u | \mathbf Q, \beta, \rho)$ be the log-likelihood function for trial $t$
- $q_i = Q_{ij} x^j$ be the value of the state $x^j$
- $v^i = e^{\beta q_i + \rho \tilde{u}_i}$ be the softmax potential
- $\eta(\boldsymbol\pi)$ be the softmax partition function.

Then we have the partial derivative of $\mathcal L$ at trial $t$ with respect to $\alpha$

$$
\partial_{\alpha} \mathcal L = \beta \Big[ \big(\mathbf u - \varsigma(\pi)\big)_i (\partial_{\alpha} Q)^i_j x^j \Big],
$$

and with respect to $\beta$

$$
\partial_{\beta} \mathcal L = u_i \Big(\mathbf I_{n_u \times n_u} - \varsigma(\boldsymbol\pi)\Big)^i_j Q_{jk} x^k
$$

and with respect to $\rho$

$$
\partial_{\rho} \mathcal L = u_i \Big(\mathbf I_{n_u \times n_u} - \varsigma(\boldsymbol\pi)\Big)^i_j \tilde{u}^j.
$$

We also compute the Hessian $\mathbf H$, defined as

$$
\mathbf H = \left[
\begin{array}{ccc}
\partial^2_{\alpha} \mathcal L & \partial_{\alpha} \partial_{\beta} \mathcal L & \partial_{\alpha} \partial_{\rho} \mathcal L \\
\partial_{\beta} \partial_{\alpha} \mathcal L & \partial^2_{\beta} \mathcal L & \partial_{\beta} \partial_{\rho} \mathcal L \\
\partial_{\rho} \partial_{\alpha} \mathcal L & \partial_{\rho} \partial_{\beta} \mathcal L & \partial^2_{\rho} \mathcal L \\
\end{array}\right].
$$

The components of $\mathbf H$ are virtually identical to that of `RWSoftmaxAgent`, with the exception of the $\partial_{\rho} \partial_{\alpha} \mathcal L$ and $\partial_{\beta} \partial_{\rho} \mathcal L$

$$
\partial^2_{\alpha} \mathcal L = \beta \Big( (\mathbf u - \varsigma(\boldsymbol\pi))_i (\partial^2_{\alpha} \mathbf Q)^i - \partial_{\alpha} \varsigma(\boldsymbol\pi)_i (\partial_{\alpha} \mathbf Q)^i \Big)_j x^j,
$$

$$
\partial^2_{\beta} \mathcal L = u_k \Bigg(\frac{(q_i q_i v^i v^i}{z^2} - \frac{q_i q_i v^i}{z} \Bigg)^k
$$

$$
\partial_{\alpha} \partial_{\beta} \mathcal L = \Bigg[ (u - \varsigma(\boldsymbol\pi)) - \beta \partial_{\beta} \varsigma(\boldsymbol\pi) \Bigg]_i (\partial_{\alpha} Q)^i_k x^k
$$

$$
\partial_{\alpha} \partial_{\rho} \mathcal L = - \beta \Big( \partial_{\boldsymbol\pi} \varsigma(\boldsymbol\pi)_i \tilde{u}^i \Big)_j (\partial_{\alpha} Q)^j_k x^k
$$

and where $\mathbf H$ is symmetric since the second derivatives of $\mathcal L$ are continuous in the neighbourhood of the parameters.

Arguments:

- **action**: `ndarray(nactions)`. One-hot action vector
- **state**: `ndarray(nstates)`. One-hot state vector

Returns:

`float`

---




### RWStickySoftmaxAgent.reset_trace

```python
fitr.agents.agents.reset_trace(self, state_only=False)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **state_only**: `bool`. If the eligibility trace is only an `nstate` dimensional vector (i.e. for a Pavlovian conditioning model) then set to `True`. For instumental models, the eligibility trace should be an `nactions` by `nstates` matrix, so keep this to `False` in that case.

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

Updates the model's parameters and computes gradients

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
fitr.agents.agents.reset_trace(self, state_only=False)
```

For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

Arguments:

- **state_only**: `bool`. If the eligibility trace is only an `nstate` dimensional vector (i.e. for a Pavlovian conditioning model) then set to `True`. For instumental models, the eligibility trace should be an `nactions` by `nstates` matrix, so keep this to `False` in that case.

---


