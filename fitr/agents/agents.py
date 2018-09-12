# -*- coding: utf-8 -*-
import autograd.numpy as np
from fitr.agents.policies import *
from fitr.agents.value_functions import *
from fitr.data import BehaviouralData

class Agent(object):
    """
    Base class for synthetic RL agents.

    Arguments:

        meta : List of metadata of arbitrary type. e.g. labels, covariates, etc.
        params : List of parameters for the agent. Should be filled for specific agent.
    """
    def __init__(self, task):
        self.task = task
        self.meta = ['Agent']
        self.params = []

    def reset_trace(self, x, u=None):
        """ For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

        Arguments:

            x: `ndarray((nstates,))` one-hot state vector
            u: `ndarray((nactions,))` one-hot action vector (optional)
        """
        if u is None:
            self.critic.etrace = np.zeros(x.size)
        else:
            self.critic.etrace = np.zeros((u.size, x.size))

    def action(self, state):
        """ Selects an action given the current state of environment.

        The implementation will vary depending on the type of agent and environment.

        Arguments:

            state: `ndarray((nstates,))` one-hot state vector
        """
        pass

    def learning(self, state, action, reward, next_state, next_action):
        """ Updates the model's parameters and computes gradients

        The implementation will vary depending on the type of agent and environment.

        Arguments:

            state: `ndarray((nstates,))` one-hot state vector
            action: `ndarray((nactions,))` one-hot action vector
            reward: scalar reward
            next_state: `ndarray((nstates,))` one-hot next-state vector
            next_action: `ndarray((nactions,))` one-hot action vector
        """
        pass

class BanditAgent(Agent):
    """ A base class for agents in bandit tasks (i.e. with one step).

    Arguments:

        task: `fitr.environments.Graph`
    """
    def __init__(self, task):
        super().__init__(task)

    def log_prob(self, state):
        """ Computes the log-likelihood over actions for a given state under the present agent parameters.

        Presently this only works for the state-action value function. In all other cases, you should define your own log-likelihood function. However, this can be used as a template.

        Arguments:

            state: `ndarray((nstates,))` one-hot state vector

        Returns:

            `ndarray((nactions,))` log-likelihood vector
        """
        return self.actor.log_prob(self.critic.Qx(state))

    def generate_data(self, ntrials):
        """ For the parent agent, this function generates data from a bandit task

        Arguments:

            ntrials: `int` number of trials

        Returns:

            `fitr.data.BehaviouralData`
        """
        data = BehaviouralData(1)
        data.add_subject(0, self.params, self.meta)
        for t in range(ntrials):
            state  = self.task.observation()
            action = self.action(state)
            next_state, reward, _ = self.task.step(action)
            data.update(0, np.hstack((state, action, reward, next_state)))
            self.learning(state, action, reward, next_state, None)
        data.make_tensor_representations()
        return data

class MDPAgent(Agent):
    """ A base class for agents that operate on MDPs.

    This mainly has implications for generating data.

    Arguments:

        task: `fitr.environments.Graph`
    """
    def __init__(self, task):
        super().__init__(task)

    def generate_data(self, ntrials):
        """ For the parent agent, this function generates data from a Markov Decision Process (MDP) task

        Arguments:

            ntrials: `int` number of trials

        Returns:

            `fitr.data.BehaviouralData`
        """
        data = BehaviouralData(1)
        data.add_subject(0, self.params, self.meta)
        for t in range(ntrials):
            done = False
            state  = self.task.observation()
            action = self.action(state)
            self.reset_trace(state, action)
            while not done:
                next_state, reward, done = self.task.step(action)
                next_action = self.action(next_state)
                data.update(0, np.hstack((state, action, reward, next_state, next_action, int(done))))
                self.learning(state, action, reward, next_state, next_action)
                state = next_state; action=next_action
        data.make_tensor_representations()
        return data

class RandomBanditAgent(BanditAgent):
    """ An agent that simply selects random actions at each trial
    """
    def __init__(self, task, **kwargs):
        super().__init__(task)
        self.meta = ['RandomAgent']
        self.critic = DummyLearner(task)
        self.actor = SoftmaxPolicy(inverse_softmax_temp=1.)
    def action(self, state):
        return self.actor.sample(self.critic.Qx(state))
    def learning(self, state, action, reward, next_state, next_action): pass

class RandomMDPAgent(MDPAgent):
    """
    An agent that simply selects random actions at each trial

    Notes
    -----
    This has been specified as an `OnPolicyAgent` arbitrarily.
    """
    def __init__(self, task, **kwargs):
        super().__init__(task)
        self.meta = ['RandomAgent']
        self.critic = DummyLearner(task)
        self.actor = SoftmaxPolicy(inverse_softmax_temp=1.)
    def action(self, state):
        return self.actor.sample(self.critic.Qx(state))
    def learning(self, state, action, reward, next_state, next_action): pass

class SARSASoftmaxAgent(MDPAgent):
    """ An agent that uses the SARSA learning rule and a softmax policy

    The softmax policy selects actions from a multinomial

    $$
    \mathbf u \sim \mathrm{Multinomial}(1, \mathbf p=\\varsigma(\mathbf v)),
    $$

    whose parameters are

    $$
    p(\mathbf u|\mathbf v) = \\varsigma(\mathbf v) = \\frac{e^{\\beta \mathbf v}}{\sum_{i}^{|\mathbf v|} e^{\\beta v_i}}.
    $$

    The value function is SARSA:

    $$
    \mathbf Q \\gets \mathbf Q + \\alpha \\big(r + \\gamma \mathbf u'^\\top \mathbf Q \mathbf x' - \mathbf u^\\top \mathbf Q \mathbf x \\big) \mathbf z,
    $$

    where $0 < \\alpha < 1$ is the learning rate, $0 \leq \\gamma \leq 1$ is a discount factor, and where the reward prediction error (RPE) is $\\delta = (r + \\gamma \mathbf u'^\\top \mathbf Q \mathbf x' - \mathbf u^\\top \mathbf Q \mathbf x)$. We have also included an eligibility trace $\mathbf z$ defined as

    $$
    \mathbf z = \mathbf u \mathbf x^\\top +  \\gamma \\lambda \mathbf z
    $$

    Arguments:

        task: `fitr.environments.Graph`
        learning_rate: Learning rate $\\alpha$
        discount_factor: Discount factor $\\gamma$
        trace_decay: Eligibility trace decay $\\lambda$
        inverse_softmax_temp: Inverse softmax temperature $\\beta$
        rng: `np.random.RandomState`

    """
    def __init__(self,
                 task,
                 learning_rate=None,
                 discount_factor=None,
                 trace_decay=None,
                 inverse_softmax_temp=None,
                 rng=np.random.RandomState()):
        super().__init__(task)
        self.meta = ['SARSASoftmax']
        if learning_rate is None: learning_rate = rng.uniform(0.01, 0.99)
        if discount_factor is None: discount_factor = rng.uniform(0.8, 1.0)
        if trace_decay is None: trace_decay = rng.uniform(0.8, 1.0)
        if inverse_softmax_temp is None: inverse_softmax_temp = rng.uniform(0.01, 10)
        self.params = [learning_rate, discount_factor, trace_decay, inverse_softmax_temp]
        self.actor  = SoftmaxPolicy(inverse_softmax_temp = inverse_softmax_temp)
        self.critic = SARSALearner(task,
                                   learning_rate=learning_rate,
                                   discount_factor=discount_factor,
                                   trace_decay=trace_decay)

    def action(self, state):
        return self.actor.sample(self.critic.Qx(state))

    def learning(self, state, action, reward, next_state, next_action):
        self.critic.update(state, action, reward, next_state, next_action)

class SARSAStickySoftmaxAgent(MDPAgent):
    """ An agent that uses the SARSA learning rule and a sticky softmax policy

    The sticky softmax policy selects actions from a multinomial

    $$
    \mathbf u \sim \mathrm{Multinomial}(1, \mathbf p=\\varsigma(\mathbf v)),
    $$

    whose parameters are

    $$
    p(\mathbf u|\mathbf v, \mathbf u_{t-1}) = \\varsigma(\mathbf v, \mathbf u_{t-1}) = \\frac{e^{\\beta \mathbf v + \\beta^\\rho \mathbf u_{t-1}}}{\sum_{i}^{|\mathbf v|} e^{\\beta v_i + \\beta^\\rho u_{t-1}^{(i)}}}.
    $$

    The value function is SARSA:

    $$
    \mathbf Q \\gets \mathbf Q + \\alpha \\big(r + \\gamma \mathbf u'^\\top \mathbf Q \mathbf x' - \mathbf u^\\top \mathbf Q \mathbf x \\big) \mathbf z,
    $$

    where $0 < \\alpha < 1$ is the learning rate, $0 \leq \\gamma \leq 1$ is a discount factor, and where the reward prediction error (RPE) is $\\delta = (r + \\gamma \mathbf u'^\\top \mathbf Q \mathbf x' - \mathbf u^\\top \mathbf Q \mathbf x)$. We have also included an eligibility trace $\mathbf z$ defined as

    $$
    \mathbf z = \mathbf u \mathbf x^\\top +  \\gamma \\lambda \mathbf z
    $$

    Arguments:

        task: `fitr.environments.Graph`
        learning_rate: Learning rate $\\alpha$
        discount_factor: Discount factor $\\gamma$
        trace_decay: Eligibility trace decay $\\lambda$
        inverse_softmax_temp: Inverse softmax temperature $\\beta$
        perseveration: Perseveration parameter $\\beta^\\rho$
        rng: `np.random.RandomState`

    """
    def __init__(self,
                 task,
                 learning_rate=None,
                 discount_factor=None,
                 trace_decay=None,
                 inverse_softmax_temp=None,
                 perseveration=None,
                 rng=np.random.RandomState()):
        super().__init__(task)
        self.meta = ['SARSAStickySoftmax']
        if learning_rate is None: learning_rate = rng.uniform(0.01, 0.99)
        if discount_factor is None: discount_factor = rng.uniform(0.8, 1.0)
        if trace_decay is None: trace_decay = rng.uniform(0.8, 1.0)
        if inverse_softmax_temp is None: inverse_softmax_temp = rng.uniform(0.01, 10)
        if perseveration is None: perseveration = rng.uniform(0.01, 10)
        self.params = [learning_rate, discount_factor, trace_decay, inverse_softmax_temp, perseveration]
        self.actor  = StickySoftmaxPolicy(inverse_softmax_temp = inverse_softmax_temp, perseveration=perseveration)
        self.critic = SARSALearner(task,
                                   learning_rate=learning_rate,
                                   discount_factor=discount_factor,
                                   trace_decay=trace_decay)

    def action(self, state):
        return self.actor.sample(self.critic.Qx(state))

    def learning(self, state, action, reward, next_state, next_action):
        self.critic.update(state, action, reward, next_state, next_action)


class QLearningSoftmaxAgent(MDPAgent):
    """ An agent that uses the Q-learning rule and a softmax policy

    The softmax policy selects actions from a multinomial

    $$
    \mathbf u \sim \mathrm{Multinomial}(1, \mathbf p=\\varsigma(\mathbf v)),
    $$

    whose parameters are

    $$
    p(\mathbf u|\mathbf v) = \\varsigma(\mathbf v) = \\frac{e^{\\beta \mathbf v}}{\sum_{i}^{|\mathbf v|} e^{\\beta v_i}}.
    $$

    The value function is Q-learning:

    $$
    \mathbf Q \\gets \mathbf Q + \\alpha \\big(r + \\gamma \max_{\mathbf u'} \mathbf u'^\\top \mathbf Q \mathbf x' - \mathbf u^\\top \mathbf Q \mathbf x \\big) \mathbf z,
    $$

    where $0 < \\alpha < 1$ is the learning rate, $0 \leq \\gamma \leq 1$ is a discount factor, and where the reward prediction error (RPE) is $\\delta = (r + \\gamma \max_{\mathbf u'} \mathbf u'^\\top \mathbf Q \mathbf x' - \mathbf u^\\top \mathbf Q \mathbf x)$. The eligibility trace $\mathbf z$ is defined as

    $$
    \mathbf z = \mathbf u \mathbf x^\\top +  \\gamma \\lambda \mathbf z
    $$

    Arguments:

        task: `fitr.environments.Graph`
        learning_rate: Learning rate $\\alpha$
        discount_factor: Discount factor $\\gamma$
        trace_decay: Eligibility trace decay $\\lambda$
        inverse_softmax_temp: Inverse softmax temperature $\\beta$
        rng: `np.random.RandomState`
    """
    def __init__(self,
                 task,
                 learning_rate=None,
                 discount_factor=None,
                 trace_decay=None,
                 inverse_softmax_temp=None,
                 rng=np.random.RandomState()):
        super().__init__(task)
        self.meta = ['QSoftmax']
        if learning_rate is None: learning_rate = rng.uniform(0.01, 0.99)
        if discount_factor is None: discount_factor = rng.uniform(0.8, 1.0)
        if trace_decay is None: trace_decay = rng.uniform(0.8, 1.0)
        if inverse_softmax_temp is None: inverse_softmax_temp = rng.uniform(0.01, 10)
        self.params = [learning_rate, discount_factor, trace_decay, inverse_softmax_temp]
        self.actor  = SoftmaxPolicy(inverse_softmax_temp = inverse_softmax_temp)
        self.critic = QLearner(task,
                               learning_rate=learning_rate,
                               discount_factor=discount_factor,
                               trace_decay=trace_decay)

    def action(self, state):
        return self.actor.sample(self.critic.Qx(state))

    def learning(self, state, action, reward, next_state, next_action):
        self.critic.update(state, action, reward, next_state, next_action)

class RWSoftmaxAgent(BanditAgent):
    """ An instrumental Rescorla-Wagner agent with a softmax policy

    The softmax policy selects actions from a multinomial

    $$
    \mathbf u \sim \mathrm{Multinomial}(1, \mathbf p=\\varsigma(\mathbf v)),
    $$

    whose parameters are

    $$
    p(\mathbf u|\mathbf v) = \\varsigma(\mathbf v) = \\frac{e^{\\beta \mathbf v}}{\sum_{i}^{|\mathbf v|} e^{\\beta v_i}}.
    $$

    The value function is the Rescorla-Wagner learning rule:

    $$
    \mathbf Q \\gets \mathbf Q + \\alpha \\big(r - \mathbf u^\\top \mathbf Q \mathbf x \\big) \mathbf u \mathbf x^\\top,
    $$

    where $0 < \\alpha < 1$ is the learning rate, $0 \leq \\gamma \leq 1$ is a discount factor, and where the reward prediction error (RPE) is $\\delta = (r - \mathbf u^\\top \mathbf Q \mathbf x)$.

    Arguments:

        task: `fitr.environments.Graph`
        learning_rate: Learning rate $\\alpha$
        inverse_softmax_temp: Inverse softmax temperature $\\beta$
        rng: `np.random.RandomState`

    """
    def __init__(self,
                 task,
                 learning_rate=None,
                 inverse_softmax_temp=None,
                 rng=np.random.RandomState()):
        super().__init__(task)
        self.meta = ['RWSoftmaxAgent']
        if learning_rate is None: learning_rate = rng.uniform(0.01, 0.99)
        if inverse_softmax_temp is None: inverse_softmax_temp = rng.uniform(0.01, 10)
        self.params = [learning_rate, inverse_softmax_temp]
        self.actor  = SoftmaxPolicy(inverse_softmax_temp = inverse_softmax_temp)
        self.critic = InstrumentalRescorlaWagnerLearner(task, learning_rate=learning_rate)

        # Storage for partial derivatives
        self.d_logprob = {
            'learning_rate': 0,
            'inverse_softmax_temp': 0
        }

    def action(self, state):
        return self.actor.sample(self.critic.Qx(state))

    def learning(self, state, action, reward, next_state, next_action):
        self.critic.update(state, action, reward, next_state, None)

    def log_prob(self, state, action):
        """ Computes the log-probability of an action taken by the agent in a given state, as well as updates all partial derivatives with respect to the parameters.

        This function overrides the `log_prob` method of the parent class.

        Arguments:

            action: `ndarray(nactions)`. One-hot action vector
            state: `ndarray(nstates)`. One-hot state vector

        Returns:

            `float`

        """
        lp = self.actor.log_prob(self.critic.Qx(x))

        # Partial derivative of log probability with respect to inverse softmax temperature
        self.d_logprob['inverse_softmax_temp'] += np.dot(action, self.actor.d_logprob['inverse_softmax_temp'])

        # Partial derivative of log probability with respect to learning rate
        #   Requires application of the chain rule
        dQ_dlr = self.critic.dQ['learning_rate']
        dq_dQ = self.critic.grad_Qx(x)
        dlp_dq = self.actor.d_logprob['action_values']
        dlp_dlr = np.dot(dlp_dq, np.sum(dq_dQ*dQ_dlr, axis=1))
        self.d_logprob['learning_rate'] += np.dot(action, dlp_dlr)


class RWStickySoftmaxAgent(BanditAgent):
    """ An instrumental Rescorla-Wagner agent with a 'sticky' softmax policy

    The softmax policy selects actions from a multinomial

    $$
    \mathbf u \sim \mathrm{Multinomial}(1, \mathbf p=\\varsigma(\mathbf v, \mathbf u_{t-1})).
    $$

    whose parameters are

    $$
    p(\mathbf u|\mathbf v, \mathbf u_{t-1}) = \\varsigma(\mathbf v, \mathbf u_{t-1}) = \\frac{e^{\\beta \mathbf v + \\beta^\\rho \mathbf u_{t-1}}}{\sum_{i}^{|\mathbf v|} e^{\\beta v_i + \\beta^\\rho u_{t-1}^{(i)}}}.
    $$

    The value function is the Rescorla-Wagner learning rule:

    $$
    \mathbf Q \\gets \mathbf Q + \\alpha \\big(r - \mathbf u^\\top \mathbf Q \mathbf x \\big) \mathbf u \mathbf x^\\top,
    $$

    where $0 < \\alpha < 1$ is the learning rate, $0 \leq \\gamma \leq 1$ is a discount factor, and where the reward prediction error (RPE) is $\\delta = (r - \mathbf u^\\top \mathbf Q \mathbf x)$.

    Arguments:

        task: `fitr.environments.Graph`
        learning_rate: Learning rate $\\alpha$
        inverse_softmax_temp: Inverse softmax temperature $\\beta$
        perseveration: Perseveration parameter $\\beta^\rho$
        rng: `np.random.RandomState`

    """
    def __init__(self,
                 task,
                 learning_rate=None,
                 inverse_softmax_temp=None,
                 perseveration=None,
                 rng=np.random.RandomState()):
        super().__init__(task)
        self.meta = ['RWSoftmaxAgent']
        if learning_rate is None: learning_rate = rng.uniform(0.01, 0.99)
        if inverse_softmax_temp is None: inverse_softmax_temp = rng.uniform(0.01, 10)
        if perseveration is None: perseveration = rng.uniform(0.01, 10)
        self.params = [learning_rate, inverse_softmax_temp, perseveration]
        self.actor  = StickySoftmaxPolicy(inverse_softmax_temp=inverse_softmax_temp, perseveration=perseveration)
        self.critic = InstrumentalRescorlaWagnerLearner(task, learning_rate=learning_rate)

    def action(self, state):
        return self.actor.sample(self.critic.Qx(state))

    def learning(self, state, action, reward, next_state, next_action):
        self.critic.update(state, action, reward, next_state, None)

class RWSoftmaxAgentRewardSensitivity(BanditAgent):
    """ An instrumental Rescorla-Wagner agent with a softmax policy, whose experienced reward is scaled by a factor $\\rho$.

    The softmax policy selects actions from a multinomial

    $$
    \mathbf u \sim \mathrm{Multinomial}(1, \mathbf p=\\varsigma(\mathbf v)),
    $$

    whose parameters are

    $$
    p(\mathbf u|\mathbf v) = \\varsigma(\mathbf v) = \\frac{e^{\\beta \mathbf v}}{\sum_{i}^{|\mathbf v|} e^{\\beta v_i}}.
    $$

    The value function is the Rescorla-Wagner learning rule with scaled reward $\\rho r$:

    $$
    \mathbf Q \\gets \mathbf Q + \\alpha \\big(\\rho r - \mathbf u^\\top \mathbf Q \mathbf x \\big) \mathbf u \mathbf x^\\top,
    $$

    where $0 < \\alpha < 1$ is the learning rate, $0 \leq \\gamma \leq 1$ is a discount factor, and where the reward prediction error (RPE) is $\\delta = (\\rho r - \mathbf u^\\top \mathbf Q \mathbf x)$.

    Arguments:

        task: `fitr.environments.Graph`
        learning_rate: Learning rate $\\alpha$
        inverse_softmax_temp: Inverse softmax temperature $\\beta$
        reward_sensitivity: Reward sensitivity parameter $\\rho$
        rng: `np.random.RandomState`

    """
    def __init__(self, task,
                 learning_rate = None,
                 inverse_softmax_temp = None,
                 reward_sensitivity = None,
                 rng = np.random.RandomState()):
        super().__init__(task)
        self.meta = ['RWSoftmaxAgent']
        if learning_rate is None: learning_rate = rng.uniform(0.01, 0.99)
        if inverse_softmax_temp is None: inverse_softmax_temp = rng.uniform(0.01, 10)
        if reward_sensitivity is None: self.rs = rng.uniform(0.01, 0.99)
        else: self.rs = reward_sensitivity
        self.params = [learning_rate, inverse_softmax_temp, self.rs]
        self.actor  = SoftmaxPolicy(inverse_softmax_temp = inverse_softmax_temp)
        self.critic = InstrumentalRescorlaWagnerLearner(task, learning_rate=learning_rate)

    def action(self, state):
        return self.actor.sample(self.critic.Qx(state))

    def learning(self, state, action, reward, next_state, next_action):
        reward *= self.rs
        self.critic.update(state, action, reward, next_state, None)
