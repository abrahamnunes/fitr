# -*- coding: utf-8 -*-
import autograd.numpy as np
import fitr.utils as fu
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
        self.logprob_ = 0

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
    def __init__(self, task, rng=np.random.RandomState(), **kwargs):
        super().__init__(task)
        self.meta = ['RandomAgent']
        self.critic = DummyLearner(task)
        self.actor = SoftmaxPolicy(inverse_softmax_temp=1., rng=rng)
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
    def __init__(self, task, rng=np.random.RandomState(), **kwargs):
        super().__init__(task)
        self.meta = ['RandomAgent']
        self.critic = DummyLearner(task)
        self.actor = SoftmaxPolicy(inverse_softmax_temp=1., rng=rng)
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
        self.actor  = SoftmaxPolicy(inverse_softmax_temp = inverse_softmax_temp,
                                    rng=rng)
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
        self.actor  = StickySoftmaxPolicy(inverse_softmax_temp = inverse_softmax_temp,
                                          perseveration=perseveration,
                                          rng=rng)
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
        self.actor  = SoftmaxPolicy(inverse_softmax_temp = inverse_softmax_temp, rng=rng)
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
        self.actor  = SoftmaxPolicy(inverse_softmax_temp = inverse_softmax_temp, rng=rng)
        self.critic = InstrumentalRescorlaWagnerLearner(task, learning_rate=learning_rate)

        # Storage for first order partial derivatives
        self.d_logprob = {
            'learning_rate': 0,
            'inverse_softmax_temp': 0
        }

        # Storage for second order partial derivatives
        self.hess_logprob = {
            'learning_rate': 0,
            'inverse_softmax_temp': 0,
            'learning_rate_inverse_softmax_temp': 0
        }

    def action(self, state):
        return self.actor.sample(self.critic.Qx(state))

    def learning(self, state, action, reward, next_state, next_action):
        self.critic.update(state, action, reward, next_state, None)

    def log_prob(self, state, action):
        """ Computes the log-probability of an action taken by the agent in a given state, as well as updates all partial derivatives with respect to the parameters.

        This function overrides the `log_prob` method of the parent class.

        Let

            - $n_u \\in \mathbb N_+$ be the dimensionality of the action space
            - $n_x \\in \mathbb N_+$ be the dimensionality of the state space
            - $\mathbf u = (u_0, u_1, u_{n_u})^\\top$ be a one-hot action vector
            - $\mathbf x = (x_0, x_1, x_{n_x})^\\top$ be a one-hot action vector
            - $\mathbf Q \\in \mathbb R^{n_u \\times n_x}$ be the state-action value function parameters
            - $\\beta \\in \mathbb R$ be the inverse softmax temperature
            - $\\alpha \\in [0, 1]$ be the learning rate
            - $\\varsigma(\\boldsymbol\\pi) = p(\mathbf u | \mathbf Q, \\beta)$ be a softmax function with logits $\\pi_i = \\beta Q_{ij} x^j$ (shown in Einstein summation convention).
            - $\mathcal L = \\log p(\mathbf u | \mathbf Q, \\beta)$ be the log-likelihood function for trial $t$
            - $q_i = Q_{ij} x^j$ be the value of the state $x^j$
            - $v^i = e^{\\beta q_i}$ be the softmax potential
            - $\\eta(\\boldsymbol\\pi)$ be the softmax partition function.

        Then we have the partial derivative of $\mathcal L$ at trial $t$ with respect to $\\alpha$

        $$
        \\partial_\\alpha \mathcal L = \\beta \Big[ \big(\mathbf u - \\varsigma(\\pi)\big)_i (\\partial_\\alpha Q)^i_j x^j \Big],
        $$

        and with respect to $\\beta$

        $$
        \\partial_\\beta \mathcal L = u_i \Big(\mathbf I_{n_u \\times n_u} - \\varsigma(\\boldsymbol\\pi)\Big)^i_j Q_{jk} x^k.
        $$

        We also compute the Hessian $\mathbf H$, defined as

        $$
        \mathbf H = \left[
            \begin{array}{cc}
            \\partial^2_{\\alpha} \mathcal L & \\partial_{\\alpha} \\partial_{\\beta} \mathcal L \\
            \\partial_{\\beta} \\partial_{\\alpha} \mathcal L & \\partial^2_{\\beta} \mathcal L \\
            \end{array}\right].
        $$

        The components of $\mathbf H$ are

        $$
        \\partial^2_{\\alpha} \mathcal L = \\beta \Big( (\mathbf u - \\varsigma(\\boldsymbol\\pi))_i (\\partial^2_\\alpha \mathbf Q)^i - \\partial_\\alpha \\varsigma(\\boldsymbol\\pi)_i (\\partial_\\alpha \mathbf Q)^i \Big)_j x^j,
        $$

        $$
        \\partial^2_{\\beta} \mathcal L = u_i \Big( \Big),
        $$

        $$
        \\partial_{\\alpha} \\partial_{\\beta} \mathcal L = \\Bigg[ (u - \\varsigma(\\boldsymbol\\pi)) - \\beta \\partial_\\beta \\varsigma(\\boldsymbol\\pi) \\Bigg]_i (\\partial_\\alpha Q)^i_k x^k.
        $$

        and where $\\partial_{\\beta} \\partial_{\\alpha} \mathcal L = \\partial_{\\alpha} \\partial_{\\beta} \mathcal L$ since the second derivatives of $\mathcal L$ are continuous in the neighbourhood of the parameters.

        Arguments:

            action: `ndarray(nactions)`. One-hot action vector
            state: `ndarray(nstates)`. One-hot state vector

        Returns:

            `float`

        """
        # Obtain the components required for computation of derivatives
        lr         = self.critic.learning_rate
        beta       = self.actor.inverse_softmax_temp
        Qx         = self.critic.Qx(state)
        logits     = beta*Qx
        pu         = fu.softmax(logits)
        du         = action - pu
        dpu_dlogit = grad.softmax(logits)
        dlogit_dB  = Qx
        dpu_dB     = np.einsum('ij,j->i', dpu_dlogit, dlogit_dB)
        D2Q_lr     = self.critic.hess_Q['learning_rate']
        DQ_lr      = self.critic.dQ['learning_rate']
        dpu_dlr    = beta*np.einsum('ij,jk,k->i', dpu_dlogit, DQ_lr, state)
        self.logprob_ += np.dot(action, self.actor.log_prob(Qx))

        # Partial derivative of log probability with respect to inverse softmax temperature
        #  Second order
        self.hess_logprob['inverse_softmax_temp'] += np.dot(action, self.actor.hess_logprob['inverse_softmax_temp'])
        #  First order
        self.d_logprob['inverse_softmax_temp'] += np.dot(action, self.actor.d_logprob['inverse_softmax_temp'])

        # Second and first partial derivative of log probability with respect to learning rate


        # Second partial derivative with respect to learning rate
        self.hess_logprob['learning_rate'] +=  beta*np.dot(np.einsum('i,ij->j', du, self.critic.hess_Q['learning_rate']) - np.einsum('i,ij->j', dpu_dlr, DQ_lr), state)

        # First partial derivative with respect to learning rate
        self.d_logprob['learning_rate'] += beta*np.einsum('i,ij,j->', du, DQ_lr, state)

        # Partial derivative with respect to both learning rate and inverse softmax
        self.hess_logprob['learning_rate_inverse_softmax_temp'] += np.einsum('i,ij,j->', du - beta*dpu_dB, DQ_lr, state)

        # Organize the gradients and hessians
        self.grad_ = np.array([self.d_logprob['learning_rate'], self.d_logprob['inverse_softmax_temp']])
        self.hess_ = np.array([[self.hess_logprob['learning_rate'], self.hess_logprob['learning_rate_inverse_softmax_temp']],
                               [self.hess_logprob['learning_rate_inverse_softmax_temp'], self.hess_logprob['inverse_softmax_temp']]])

    def _log_prob_noderivatives(self, state, action):
        """ Computes the log-probability of an action taken by the agent in a given state without updating partial derivatives with respect to the parameters.

        This function is only implemented for purposes of unit testing for comparison with `autograd` package.

        Arguments:

            action: `ndarray(nactions)`. One-hot action vector
            state: `ndarray(nstates)`. One-hot state vector

        """
        self.logprob_ += np.dot(action, self.actor._log_prob_noderivatives(self.critic.Qx(state)))

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
        self.actor  = StickySoftmaxPolicy(inverse_softmax_temp=inverse_softmax_temp,
                                          perseveration=perseveration,
                                          rng=rng)
        self.actor.a_last = np.zeros(self.task.nactions)
        self.critic = InstrumentalRescorlaWagnerLearner(task, learning_rate=learning_rate)

        # Storage for partial derivatives
        self.d_logprob = {
            'learning_rate': 0,
            'inverse_softmax_temp': 0,
            'perseveration': 0
        }

        # Storage for second order partial derivatives
        self.hess_logprob = {
            'learning_rate': 0,
            'inverse_softmax_temp': 0,
            'perseveration': 0,
            'learning_rate_inverse_softmax_temp': 0,
            'learning_rate_perseveration': 0,
            'inverse_softmax_temp_perseveration': 0
        }

    def action(self, state):
        return self.actor.sample(self.critic.Qx(state))

    def learning(self, state, action, reward, next_state, next_action):
        self.critic.update(state, action, reward, next_state, None)

    def log_prob(self, state, action):
        """ Computes the log-probability of an action taken by the agent in a given state, as well as updates all partial derivatives with respect to the parameters.

        This function overrides the `log_prob` method of the parent class.

        Let

            - $n_u \\in \mathbb N_+$ be the dimensionality of the action space
            - $n_x \\in \mathbb N_+$ be the dimensionality of the state space
            - $\mathbf u = (u_0, u_1, u_{n_u})^\\top$ be a one-hot action vector
            - $\\tilde{\mathbf u}$ be a one-hot vector representing the last trial's action, where at trial 0, $\\tilde{\mathbf u}} = \mathbf 0$.
            - $\mathbf x = (x_0, x_1, x_{n_x})^\\top$ be a one-hot action vector
            - $\mathbf Q \\in \mathbb R^{n_u \\times n_x}$ be the state-action value function parameters
            - $\\beta \\in \mathbb R$ be the inverse softmax temperature scaling the action values
            - $\\rho \\in \\mathbb R$ be the inverse softmax temperature scaling the influence of the past trial's action
            - $\\alpha \\in [0, 1]$ be the learning rate
            - $\\varsigma(\\boldsymbol\\pi) = p(\mathbf u | \mathbf Q, \\beta, \\rho)$ be a softmax function with logits $\\pi_i = \\beta Q_{ij} x^j + \\rho \\tilde{u}_i$ (shown in Einstein summation convention).
            - $\mathcal L = \\log p(\mathbf u | \mathbf Q, \\beta, \\rho)$ be the log-likelihood function for trial $t$
            - $q_i = Q_{ij} x^j$ be the value of the state $x^j$
            - $v^i = e^{\\beta q_i + \\rho \\tilde{u}_i}$ be the softmax potential
            - $\\eta(\\boldsymbol\\pi)$ be the softmax partition function.

        Then we have the partial derivative of $\mathcal L$ at trial $t$ with respect to $\\alpha$

        $$
        \\partial_\\alpha \mathcal L = \\beta \Big[ \big(\mathbf u - \\varsigma(\\pi)\big)_i (\\partial_\\alpha Q)^i_j x^j \Big],
        $$

        and with respect to $\\beta$

        $$
        \\partial_\\beta \mathcal L = u_i \Big(\mathbf I_{n_u \\times n_u} - \\varsigma(\\boldsymbol\\pi)\Big)^i_j Q_{jk} x^k
        $$

        and with respect to $\\rho$

        $$
        \\partial_\\rho \mathcal L = u_i \Big(\mathbf I_{n_u \\times n_u} - \\varsigma(\\boldsymbol\\pi)\Big)^i_j \\tilde{u}^j.
        $$

        We also compute the Hessian $\mathbf H$, defined as

        $$
        \mathbf H = \left[
            \begin{array}{ccc}
            \\partial^2_{\\alpha} \mathcal L & \\partial_{\\alpha} \\partial_{\\beta} \mathcal L & \\partial_{\\alpha} \\partial_{\\rho} \mathcal L \\
            \\partial_{\\beta} \\partial_{\\alpha} \mathcal L & \\partial^2_{\\beta} \mathcal L & \\partial_{\\beta} \\partial_{\\rho} \mathcal L \\
            \\partial_{\\rho} \\partial_{\\alpha} \mathcal L & \\partial_{\\rho} \\partial_{\\beta} \mathcal L & \\partial^2_{\\rho} \mathcal L \\
            \end{array}\right].
        $$

        The components of $\mathbf H$ are virtually identical to that of `RWSoftmaxAgent`, with the exception of the $\\partial_{\\rho} \\partial_{\\alpha} \mathcal L$ and $\\partial_{\\beta} \\partial_{\\rho} \mathcal L$

        $$
        \\partial^2_{\\alpha} \mathcal L = \\beta \Big( (\mathbf u - \\varsigma(\\boldsymbol\\pi))_i (\\partial^2_\\alpha \mathbf Q)^i - \\partial_\\alpha \\varsigma(\\boldsymbol\\pi)_i (\\partial_\\alpha \mathbf Q)^i \Big)_j x^j,
        $$

        $$
        \\partial^2_{\\beta} \mathcal L = u_k \\Bigg(\\frac{(q_i q_i v^i v^i}{z^2} - \\frac{q_i q_i v^i}{z} \\Bigg)^k
        $$

        $$
        \\partial_{\\alpha} \\partial_{\\beta} \mathcal L = \\Bigg[ (u - \\varsigma(\\boldsymbol\\pi)) - \\beta \\partial_\\beta \\varsigma(\\boldsymbol\\pi) \\Bigg]_i (\\partial_\\alpha Q)^i_k x^k
        $$

        $$
        \\partial_{\\alpha} \\partial_{\\rho} \mathcal L = - \\beta \\Big( \\partial_\\boldsymbol\\pi \\varsigma(\\boldsymboll\\pi)_i \\tilde{u}^i \\Big)_j (\\partial_\\alpha Q)^j_k x^k
        $$

        and where $\\mathbf H$ is symmetric since the second derivatives of $\mathcal L$ are continuous in the neighbourhood of the parameters.

        Arguments:

            action: `ndarray(nactions)`. One-hot action vector
            state: `ndarray(nstates)`. One-hot state vector

        Returns:

            `float`

        """
        # Obtain the components required for computation of derivatives
        lr             = self.critic.learning_rate
        beta           = self.actor.inverse_softmax_temp
        persev         = self.actor.perseveration
        Qx             = self.critic.Qx(state)
        logits         = beta*Qx + persev*self.actor.a_last
        pu             = fu.softmax(logits)
        du             = action - pu
        dpu_dlogit     = grad.softmax(logits)
        dlogit_dB      = Qx
        dlogit_drho    = self.actor.a_last
        dpu_dB         = np.einsum('ij,j->i', dpu_dlogit, dlogit_dB)
        dpu_drho       = np.einsum('ij,j->i', dpu_dlogit, dlogit_drho)
        D2Q_lr         = self.critic.hess_Q['learning_rate']
        DQ_lr          = self.critic.dQ['learning_rate']
        dpu_dlr        = beta*np.einsum('ij,jk,k->i', dpu_dlogit, DQ_lr, state)

        # Compute the log-probability
        self.logprob_ += np.dot(action, self.actor.log_prob(Qx))
        self.actor.a_last = action

        # Partial derivative of log probability with respect to inverse softmax temperature
        #  Second order
        self.hess_logprob['inverse_softmax_temp'] += np.dot(action, self.actor.hess_logprob['inverse_softmax_temp'])
        #  First order
        self.d_logprob['inverse_softmax_temp'] += np.dot(action, self.actor.d_logprob['inverse_softmax_temp'])

        # Partial derivative of log probability with respect to perseveration
        #  Second order
        self.hess_logprob['perseveration'] += np.dot(action, self.actor.hess_logprob['perseveration'])
        # First order
        self.d_logprob['perseveration'] += np.dot(action, self.actor.d_logprob['perseveration'])

        # Second partial derivative with respect to learning rate
        self.hess_logprob['learning_rate'] +=  beta*np.dot(np.einsum('i,ij->j', du, D2Q_lr) - np.einsum('i,ij->j', dpu_dlr, DQ_lr), state)

        # First partial derivative with respect to learning rate
        self.d_logprob['learning_rate'] += beta*np.einsum('i,ij,j->', du, DQ_lr, state)

        # Partial derivative with respect to both learning rate and inverse softmax
        self.hess_logprob['learning_rate_inverse_softmax_temp'] += np.einsum('i,ij,j->', du - beta*dpu_dB, DQ_lr, state)

        # Partial derivative with respect to both learning rate and perseveration
        self.hess_logprob['learning_rate_perseveration'] += -beta*np.einsum('i,ij,j->', dpu_drho, DQ_lr, state)

        # Partial derivative with respect to both inverse softmax temperature and perseveration
        self.hess_logprob['inverse_softmax_temp_perseveration'] += np.dot(action, self.actor.hess_logprob['inverse_softmax_temp_perseveration'])

        # Organize the gradients and hessians
        self.grad_ = np.array([self.d_logprob['learning_rate'],
                               self.d_logprob['inverse_softmax_temp'],
                               self.d_logprob['perseveration']])

        self.hess_ = np.array([[self.hess_logprob['learning_rate'],
                                self.hess_logprob['learning_rate_inverse_softmax_temp'],
                                self.hess_logprob['learning_rate_perseveration']],

                               [self.hess_logprob['learning_rate_inverse_softmax_temp'],
                               self.hess_logprob['inverse_softmax_temp'],
                               self.hess_logprob['inverse_softmax_temp_perseveration']],

                               [self.hess_logprob['learning_rate_perseveration'],
                                self.hess_logprob['inverse_softmax_temp_perseveration'],
                                self.hess_logprob['perseveration']]])


    def _log_prob_noderivatives(self, state, action):
        """ Computes the log-probability of an action taken by the agent in a given state without updating partial derivatives with respect to the parameters.

        This function is only implemented for purposes of unit testing for comparison with `autograd` package.

        Arguments:

            action: `ndarray(nactions)`. One-hot action vector
            state: `ndarray(nstates)`. One-hot state vector

        """
        self.logprob_ += np.dot(action, self.actor._log_prob_noderivatives(self.critic.Qx(state)))
        self.actor.a_last = action

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
