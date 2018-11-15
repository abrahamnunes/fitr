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

    def reset_trace(self, state_only=False):
        """ For agents with eligibility traces, this resets the eligibility trace (for episodic tasks)

        Arguments:

            state_only: `bool`. If the eligibility trace is only an `nstate` dimensional vector (i.e. for a Pavlovian conditioning model) then set to `True`. For instumental models, the eligibility trace should be an `nactions` by `nstates` matrix, so keep this to `False` in that case.

        """
        if state_only is True:
            self.critic.etrace = np.zeros(self.task.nstates)
        else:
            self.critic.etrace = np.zeros((self.task.nactions, self.task.nstates))

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

    def generate_data(self, ntrials, state_only=False):
        """ For the parent agent, this function generates data from a Markov Decision Process (MDP) task

        Arguments:

            ntrials: `int` number of trials
            state_only: `bool`. If the eligibility trace is only an `nstate` dimensional vector (i.e. for a Pavlovian conditioning model) then set to `True`. For instumental models, the eligibility trace should be an `nactions` by `nstates` matrix, so keep this to `False` in that case.

        Returns:

            `fitr.data.BehaviouralData`
        """
        data = BehaviouralData(1)
        data.add_subject(0, self.params, self.meta)
        for t in range(ntrials):
            done = False
            state  = self.task.observation()
            action = self.action(state)
            self.reset_trace(state_only=state_only)
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

        self.grad_ = None
        self.hess_ = None

        self.d_logprob = {
            'learning_rate': 0,
            'discount_factor': 0,
            'trace_decay': 0,
            'inverse_softmax_temp': 0,
            'Q': np.zeros(self.critic.Q.shape)
        }

        self.d_action_values = {
            'Q': None,
            'learning_rate': None,
            'discount_factor': None,
            'trace_decay': None
        }

        self.hess_logprob = {
            'learning_rate': 0,
            'learning_rate_inverse_softmax_temp': 0,
            'learning_rate_discount_factor': 0,
            'learning_rate_trace_decay': 0,
            'inverse_softmax_temp': 0,
            'inverse_softmax_temp_discount_factor': 0,
            'inverse_softmax_temp_trace_decay': 0,
            'discount_factor_trace_decay': 0,
	    'discount_factor': 0,
            'trace_decay': 0
        }

    def action(self, state):
        return self.actor.sample(self.critic.Qx(state))

    def learning(self, state, action, reward, next_state, next_action):
        self.critic.update(state, action, reward, next_state, next_action)

    def log_prob(self, state, action):
        """ Computes the log-probability of the given action and state under the model, while also computing first and second order derivatives.

        This model has four free parameters:

    	- Learning rate $\\alpha$
        - Inverse softmax temperature $\\beta$
        - Discount factor $\\gamma$
        - Trace decay $\\lambda$

        __First-order partial derivatives__

        We can break down the computation using the chain rule to reuse previously computed derivatives:

        $$
        \\pd{\\cL}{\\alpha}  = \\pd{\\cL}{\\logits} \\pd{\\logits}{\\mathbf q} \\pd{\\mathbf q}{\\mathbf Q} \\pd{\\mathbf Q}{\\alpha}
        $$

        $$
        \\pd{\\cL}{\\beta}   = \\pd{\\cL}{\\logits} \\pd{\\logits}{\\beta}
        $$

        $$
        \\pd{\\cL}{\\gamma}  = \\pd{\\cL}{\\logits} \\pd{\\logits}{\\mathbf q} \\pd{\\mathbf q}{\\mathbf Q} \\pd{\\mathbf Q}{\\gamma}
        $$

        $$
        \\pd{\\cL}{\\lambda} = \\pd{\\cL}{\\logits} \\pd{\\logits}{\\mathbf q} \\pd{\\mathbf q}{\\mathbf Q} \\pd{\\mathbf Q}{\\lambda}
        $$

        _Action Probabilities_

        $$
        \\partial_\\alpha \\varsigma = \\pd{\\varsigma}{\\logits} \\pd{\\logits}{\\mathbf q} \\pd{\\mathbf q}{\\mathbf Q} \\big( \\partial_\\alpha \\mathbf Q \\big) = \\beta \\big(\\partial_{\\logits} \\varsigma \\big)_i \\big( \\partial_\\alpha Q \\big)^i_j x^j
        $$

        _Value Function_

        $$
        \\partial_\\alpha Q_{ij} =  \\partial_\\alpha Q_{ij} + (\\delta + \\alpha \\partial_\\alpha \\delta) z_{ij}
        $$

        $$
        \\partial_\\gamma Q_{ij} =  \\partial_\\gamma Q_{ij} + \\alpha \\big( (\\partial_\\gamma \\delta) z_{ij} + \\delta (\\partial_\\gamma z_{ij}) \\big)
        $$

        $$
        \\partial_\\lambda Q_{ij} =  \\partial_\\lambda Q_{ij} + \\alpha \\big( (\\partial_\\lambda \\delta) z_{ij} + \\delta (\\partial_\\lambda z_{ij}) \\big)
        $$

        _Reward Prediction Error_

        $$
        \\partial_\\alpha \\delta = (\\partial_{\\mathbf Q} \\delta)_{ij} (\\partial_\\alpha Q)^{ij}
        $$

        $$
        \\partial_\\gamma \\delta = (\\partial_{\\mathbf Q} \\delta)_{ij} (\\partial_\\gamma Q)^{ij} + \\tilde{u}_i Q^i_j \\tilde{x}^j
        $$

        $$
        \\partial_\\lambda \\delta = (\\partial_{\\mathbf Q} \\delta)_{ij} (\\partial_\\lambda Q)^{ij}
        $$

        _Trace Decay_

        $$
        \\partial_\\gamma z_{ij} = \\lambda \\big(z_{ij} + \\gamma (\\partial_\\gamma z_{ij}) \\big)
        $$

        $$
        \\partial_\\lambda z_{ij} = \\gamma \\big(z_{ij} + \\lambda (\\partial_\\lambda z_{ij}) \\big)
        $$

        _Simplified Components of the Gradient Vector_

        $$
        \\pd{\\cL}{\\alpha}  = \\beta \\big[\\mathbf u - \\varsigma(\\logits) \\big]_i \\big( \\partial_\\alpha Q \\big)^i_j x^j   = \\beta \\big[ u_i (\\partial_\\alpha Q)^i_j x^j - p(u_i) (\\partial_\\alpha Q)^i_j x^j \\big]
        $$

        $$
        \\pd{\\cL}{\\beta}   =  \\big[\\mathbf u - \\varsigma(\\logits)\\big]_i Q^i_j x^j = u_i Q^i_j x^j - p(u_i) Q^i_j x^j
        $$

        $$
        \\pd{\\cL}{\\gamma}  = \\beta \\big[\\mathbf u - \\varsigma(\\logits) \\big]_i \\big( \\partial_\\gamma Q \\big)^i_j x^j
        $$

        $$
        \\pd{\\cL}{\\lambda} = \\beta \\big[\\mathbf u - \\varsigma(\\logits) \\big]_i \\big( \\partial_\\lambda Q \\big)^i_j x^j
        $$

        __Second-Order Partial Derivatives__

        The Hessian matrix for this model is

        $$
        \\mathbf H = \\left[
        \\begin{array}{cccc}
        \\pHd{\\cL}{\\alpha} 		   & \\pHo{\\cL}{\\alpha}{\\beta}  & \\pHo{\\cL}{\\alpha}{\\gamma}  & \\pHo{\\cL}{\\alpha}{\\lambda} \\\\
        \\pHo{\\cL}{\\beta}{\\alpha}   & \\pHd{\\cL}{\\beta} 		   & \\pHo{\\cL}{\\beta}{\\gamma}   & \\pHo{\\cL}{\\beta}{\\lambda}  \\\\
        \\pHo{\\cL}{\\gamma}{\\alpha}  & \\pHo{\\cL}{\\gamma}{\\beta}  & \\pHd{\\cL}{\\gamma} 		    & \\pHo{\\cL}{\\gamma}{\\lambda} \\\\
        \\pHo{\\cL}{\\lambda}{\\alpha} & \\pHo{\\cL}{\\lambda}{\\beta} & \\pHo{\\cL}{\\lambda}{\\gamma} & \\pHd{\\cL}{\\lambda} 		 \\\\
        \\end{array}\\right],
        $$

        where the second-order partial derivatives are such that $\\mathbf H$ is symmetrical. We must therefore compute 10 second order partial derivatives, shown below:

        $$
        \\pHd{\\cL}{\\alpha} = \\beta \\Big[ (\\mathbf u - \\varsigma(\\logits))_i \\big( \\partial^2_\\alpha Q \\big)^i - \\big( \\partial_\\alpha \\varsigma \\big)_j \\big( \\partial_\\alpha Q)^j_k x^k \\Big]_l x^l
        $$

        $$
        \\pHd{\\cL}{\\beta} = \\Big( q_i \\varsigma(\\logits)^i \\Big)^2 - \\mathbf q \\odot \\mathbf q \\odot \\varsigma(\\logits)
        $$

        $$
        \\pHd{\\cL}{\\gamma}  = \\beta \\Big[ (\\mathbf u - \\varsigma(\\logits))_i \\big( \\partial^2_\\gamma Q \\big)^i - \\big( \\partial_\\gamma \\varsigma \\big)_j \\big( \\partial_\\gamma Q)^j_k x^k \\Big]_l x^l
        $$

        $$
        \\pHd{\\cL}{\\lambda}  = \\beta \\Big[ (\\mathbf u - \\varsigma(\\logits))_i \\big( \\partial^2_\\lambda Q \\big)^i - \\big( \\partial_\\lambda \\varsigma \\big)_j \\big( \\partial_\\lambda Q)^j_k x^k \\Big]_l x^l
        $$

        The off diagonal elements of the Hessian are as follows:

        $$
        \\pHo{\\cL}{\\alpha}{\\beta}   = \\bigg(\\mathbf u - \\varsigma(\\logits) - \\beta \\big(\\partial_\\beta \\varsigma \\big) \\bigg)_i \\big( \\partial_\\alpha Q \\big)^i_j x^j
        $$

        $$
        \\pHo{\\cL}{\\beta}{\\gamma}   =  \\bigg(\\mathbf u - \\varsigma(\\logits) - \\beta \\big(\\partial_\\beta \\varsigma \\big) \\bigg)_i \\big( \\partial_\\gamma Q \\big)^i_j x^j
        $$

        $$
        \\pHo{\\cL}{\\beta}{\\lambda}  =  \\bigg(\\mathbf u - \\varsigma(\\logits) - \\beta \\big(\\partial_\\beta \\varsigma \\big) \\bigg)_i \\big( \\partial_\\lambda Q \\big)^i_j x^j
        $$

        $$
        \\pHo{\\cL}{\\alpha}{\\gamma}  = \\beta \\Big((\\mathbf u - \\varsigma(\\logits))_i \\big(\\partial_\\alpha \\partial_\\gamma Q \\big)^i - \\big( \\partial_\\gamma \\varsigma \\big)_j \\big( \\partial_\\alpha Q \\big)^j \\Big)_k x^k
        $$

        $$
        \\pHo{\\cL}{\\alpha}{\\lambda} =  \\beta \\Big((\\mathbf u - \\varsigma(\\logits))_i \\big(\\partial_\\alpha \\partial_\\lambda Q \\big)^i - \\big( \\partial_\\lambda \\varsigma \\big)_j \\big( \\partial_\\alpha Q \\big)^j \\Big)_k x^k
        $$

        $$
        \\pHo{\\cL}{\\gamma}{\\lambda} =  \\beta \\Big((\\mathbf u - \\varsigma(\\logits))_i \\big(\\partial_\\lambda \\partial_\\gamma Q \\big)^i - \\big( \\partial_\\lambda \\varsigma \\big)_j \\big( \\partial_\\gamma Q \\big)^j \\Big)_k x^k
        $$

        _Reward Prediction Error_

        $$
        \\partial^2_\\alpha \\delta = (\\partial_{\\mathbf Q} \\delta)_{ij} (\\partial^2_\\alpha Q)^{ij}
        $$

        $$
        \\partial^2_\\gamma \\delta = (\\partial_{\\mathbf Q} \\delta)_{ij} (\\partial^2_\\gamma Q)^{ij} + 2 \\tilde{u}_i \\big(\\partial_\\gamma Q\\big)^i_j \\tilde{x}^j
        $$

        $$
        \\partial^2_\\lambda \\delta = (\\partial_{\\mathbf Q} \\delta)_{ij} (\\partial^2_\\lambda Q)^{ij}
        $$

        $$
        \\partial_\\alpha \\partial_\\gamma \\delta = (\\partial_{\\mathbf Q} \\delta)_{ij} (\\partial_\\gamma \\partial_\\alpha Q)^{ij} + \\tilde{u}_i \\big(\\partial_\\alpha Q\\big)^i_j \\tilde{x}^j
        $$

        $$
        \\partial_\\alpha \\partial_\\lambda \\delta = (\\partial_{\\mathbf Q} \\delta)_{ij} (\\partial_\\alpha \\partial_\\lambda Q)^{ij}
        $$

        $$
        \\partial_\\gamma \\partial_\\lambda \\delta = (\\partial_{\\mathbf Q} \\delta)_{ij} (\\partial_\\gamma \\partial_\\lambda Q)^{ij} + \\tilde{u}_i \\big(\\partial_\\lambda Q\\big)^i_j \\tilde{x}^j
        $$

        _Value Function_

        $$
        \\partial^2_\\alpha Q_{ij} = \\partial^2_\\alpha Q_{ij} + 2(\\partial_\\alpha \\delta) z_{ij} + \\alpha (\\partial^2_\\alpha \\delta) z_{ij}
        $$

        $$
        \\partial^2_\\gamma Q_{ij} = \\partial^2_\\gamma Q_{ij} +  \\alpha \\Big( \\big(\\partial^2_\\gamma \\delta \\big)z_{ij} +  \\big(\\partial_\\gamma \\delta \\big) \\big(\\partial_\\gamma z_{ij}\\big) + \\big(\\partial_\\gamma \\delta \\big) \\big(\\partial^2_\\gamma z_{ij}\\big) \\Big)
        $$

        $$
        \\partial^2_\\lambda Q_{ij} = \\partial^2_\\lambda Q_{ij} +  \\alpha \\Big( \\big(\\partial^2_\\lambda \\delta \\big)z_{ij} +  \\big(\\partial_\\lambda \\delta \\big) \\big(\\partial_\\lambda z_{ij}\\big) + \\big(\\partial_\\lambda \\delta \\big) \\big(\\partial^2_\\lambda z_{ij}\\big) \\Big)
        $$

        $$
        \\partial_\\alpha \\partial_\\gamma Q_{ij} =  \\partial_\\alpha \\partial_\\gamma Q_{ij} + (\\partial_\\gamma \\delta) z_{ij} + \\delta \\big(\\partial_\\gamma z_{ij} \\big) + \\alpha(\\partial_\\alpha \\delta)\\big(\\partial_\\gamma z_{ij} \\big) + \\alpha(\\partial_\\alpha \\partial_\\gamma \\delta) z_{ij}
        $$

        $$
        \\partial_\\alpha \\partial_\\lambda Q_{ij} = \\partial_\\alpha \\partial_\\lambda Q_{ij} + (\\partial_\\lambda \\delta) z_{ij} + \\delta \\big(\\partial_\\lambda z_{ij} \\big) + \\alpha(\\partial_\\alpha \\delta)\\big(\\partial_\\lambda z_{ij} \\big) + \\alpha(\\partial_\\alpha \\partial_\\lambda \\delta) z_{ij}
        $$

        $$
        \\partial_\\gamma \\partial_\\lambda Q_{ij} = \\partial_\\gamma \\partial_\\lambda Q_{ij} + \\alpha \\Big[ \\big( \\partial_\\lambda \\partial_\\gamma \\delta \\big) z_{ij} + \\big( \\partial_\\gamma \\delta \\big)\\big(\\partial_\\lambda z_{ij} \\big) + \\big(\\partial_\\lambda \\delta \\big)\\big(\\partial_\\gamma z_{ij} \\big) + \\delta \\big(\\partial_\\lambda \\partial_\\gamma z_{ij} \\big) \\Big]
        $$

        _Trace Decay_

        $$
        \\partial^2_\\gamma z = \\lambda \\Big( 2\\big(\\partial_\\gamma z\\big) + \\gamma \\big(\\partial^2_\\gamma z \\big) \\Big)
        $$

        $$
        \\partial^2_\\lambda z = \\gamma \\Big( 2\\big(\\partial_\\lambda z\\big) + \\lambda \\big(\\partial^2_\\lambda z \\big) \\Big)
        $$

        $$
        \\partial_\\gamma \\partial_\\lambda z = z  + \\gamma \\big( \\partial_\\gamma z \\big) + \\lambda \\big( \\partial_\\lambda z \\big) + \\lambda \\gamma \\big( \\partial_\\gamma \\partial_\\lambda z \\big)
        $$

        Arguments:

            action: `ndarray(nactions)`. One-hot action vector
            state: `ndarray(nstates)`. One-hot state vector

        """
        lr = self.critic.learning_rate
        beta = self.actor.inverse_softmax_temp
        q  = self.critic.Qx(state)
        self.logprob_ += np.dot(action, self.actor.log_prob(q))
        logits=beta*q
        Dq_Q = state
        Dlogit_B = q
        Dlogit_q = beta
        Dlp_logit = np.eye(action.size) - np.tile(fu.softmax(logits).flatten(), [logits.size, 1])
        pu = self.actor.action_prob(q)
        du = action - pu
        dpu_logit = grad.softmax(logits)
        DQ_lr_state = np.einsum('ij,j->i', self.critic.dQ['learning_rate'], state)
        DQ_dc_state = np.einsum('ij,j->i', self.critic.dQ['discount_factor'], state)
        DQ_td_state = np.einsum('ij,j->i', self.critic.dQ['trace_decay'], state)
        dpu_lr = beta*np.einsum('ij,j->i', dpu_logit, DQ_lr_state)
        dpu_B  =      np.einsum('ij,j->i', dpu_logit, Dlogit_B)
        dpu_dc = beta*np.einsum('ij,j->i', dpu_logit, DQ_dc_state)
        dpu_td = beta*np.einsum('ij,j->i', dpu_logit, DQ_td_state)

        # Second order derivatives
        self.hess_logprob['learning_rate']   += beta*np.dot(np.einsum('i,ij->j', du, self.critic.hess_Q['learning_rate'])   - np.einsum('i,ij->j', dpu_lr, self.critic.dQ['learning_rate']), state)
        self.hess_logprob['discount_factor'] += beta*np.dot(np.einsum('i,ij->j', du, self.critic.hess_Q['discount_factor']) - np.einsum('i,ij->j', dpu_dc, self.critic.dQ['discount_factor']), state)
        self.hess_logprob['trace_decay']     += beta*np.dot(np.einsum('i,ij->j', du, self.critic.hess_Q['trace_decay'])     - np.einsum('i,ij->j', dpu_td, self.critic.dQ['trace_decay']), state)

        self.hess_logprob['learning_rate_discount_factor'] += beta*np.dot(np.einsum('i,ij->j', du, self.critic.hess_Q['learning_rate_discount_factor']) - np.einsum('i,ij->j', dpu_dc, self.critic.dQ['learning_rate']), state)
        self.hess_logprob['learning_rate_trace_decay']     += beta*np.dot(np.einsum('i,ij->j', du, self.critic.hess_Q['learning_rate_trace_decay'])     - np.einsum('i,ij->j', dpu_td, self.critic.dQ['learning_rate']), state)
        self.hess_logprob['discount_factor_trace_decay']   += beta*np.dot(np.einsum('i,ij->j', du, self.critic.hess_Q['discount_factor_trace_decay'])   - np.einsum('i,ij->j', dpu_td, self.critic.dQ['discount_factor']), state)

        self.hess_logprob['inverse_softmax_temp']                 += np.dot(action, self.actor.hess_logprob['inverse_softmax_temp'])
        self.hess_logprob['learning_rate_inverse_softmax_temp']   += np.dot(du - beta*dpu_B, DQ_lr_state)
        self.hess_logprob['inverse_softmax_temp_discount_factor'] += np.dot(du - beta*dpu_B, DQ_dc_state)
        self.hess_logprob['inverse_softmax_temp_trace_decay']     += np.dot(du - beta*dpu_B, DQ_td_state)

        # Derivatives of log-probability with respect to inverse softmax temperature
        #   First order
        self.d_logprob['inverse_softmax_temp'] += np.dot(action, self.actor.d_logprob['inverse_softmax_temp'])
        self.d_logprob['Q'] = Dlp_logit*Dlogit_q

        # Derivatives of log-probability with respect to learning rate
        #   First order
        self.d_action_values['learning_rate'] = np.dot(self.critic.dQ['learning_rate'], Dq_Q)
        self.d_logprob['learning_rate']      += np.einsum('i,ij,j->', action, self.actor.d_logprob['action_values'], self.d_action_values['learning_rate'])

        # Derivatives of log-probability with respect to discount factor
        #   First order
        self.d_action_values['discount_factor'] = np.dot(self.critic.dQ['discount_factor'], Dq_Q)
        self.d_logprob['discount_factor'] += np.einsum('i,ij,j->', action, self.actor.d_logprob['action_values'], self.d_action_values['discount_factor'])

        # Derivatives of log-probability with respect to trace decay
        #   First order
        self.d_action_values['trace_decay'] = np.dot(self.critic.dQ['trace_decay'], Dq_Q)
        self.d_logprob['trace_decay'] += np.einsum('i,ij,j->', action, self.actor.d_logprob['action_values'], self.d_action_values['trace_decay'])

        # Organize the gradient and hessian
        self.grad_ = np.array([self.d_logprob['learning_rate'],
                               self.d_logprob['inverse_softmax_temp'],
                               self.d_logprob['discount_factor'],
                               self.d_logprob['trace_decay']])

        self.hess_ = np.array([[self.hess_logprob['learning_rate'],
                                self.hess_logprob['learning_rate_inverse_softmax_temp'],
                                self.hess_logprob['learning_rate_discount_factor'],
                                self.hess_logprob['learning_rate_trace_decay']],

                               [self.hess_logprob['learning_rate_inverse_softmax_temp'],
                                self.hess_logprob['inverse_softmax_temp'],
                                self.hess_logprob['inverse_softmax_temp_discount_factor'],
                                self.hess_logprob['inverse_softmax_temp_trace_decay']],

                               [self.hess_logprob['learning_rate_discount_factor'],
                                self.hess_logprob['inverse_softmax_temp_discount_factor'],
                                self.hess_logprob['discount_factor'],
                                self.hess_logprob['discount_factor_trace_decay']],

                               [self.hess_logprob['learning_rate_trace_decay'],
                                self.hess_logprob['inverse_softmax_temp_trace_decay'],
                                self.hess_logprob['discount_factor_trace_decay'],
                                self.hess_logprob['trace_decay']]])

    def _log_prob_noderivatives(self, state, action):
        """ Computes the log-probability of the given action and state under the model without computing derivatives.

        This is here only to facilitate testing our gradient/hessian computations against autograd

        Arguments:

            action: `ndarray(nactions)`. One-hot action vector
            state: `ndarray(nstates)`. One-hot state vector

        """
        Qx             = self.critic.Qx(state)
        self.logprob_ += np.dot(action, self.actor._log_prob_noderivatives(Qx))

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

        self.grad_ = None
        self.hess_ = None

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
        \\partial_{\\alpha} \mathcal L = \\beta \Big[ \\big(\mathbf u - \\varsigma(\\pi)\\big)_i (\\partial_{\\alpha} Q)^i_j x^j \Big],
        $$

        and with respect to $\\beta$

        $$
        \\partial_{\\beta} \mathcal L = u_i \Big(\mathbf I_{n_u \\times n_u} - \\varsigma(\\boldsymbol\\pi)\Big)^i_j Q_{jk} x^k.
        $$

        We also compute the Hessian $\mathbf H$, defined as

        $$
        \mathbf H = \left[
            \\begin{array}{cc}
            \\partial^2_{\\alpha} \mathcal L & \\partial_{\\alpha} \\partial_{\\beta} \mathcal L \\\\
            \\partial_{\\beta} \\partial_{\\alpha} \mathcal L & \\partial^2_{\\beta} \mathcal L \\\\
            \\end{array}\\right].
        $$

        The components of $\mathbf H$ are

        $$
        \\partial^2_{\\alpha} \mathcal L = \\beta \Big( (\mathbf u - \\varsigma(\\boldsymbol\\pi))_i (\\partial^2_\\alpha \mathbf Q)^i - \\partial_{\\alpha} \\varsigma(\\boldsymbol\\pi)_i (\\partial_{\\alpha} \mathbf Q)^i \Big)_j x^j,
        $$

        $$
        \\partial^2_{\\beta} \mathcal L = u_i \Big( \Big),
        $$

        $$
        \\partial_{\\alpha} \\partial_{\\beta} \mathcal L = \\Bigg[ (u - \\varsigma(\\boldsymbol\\pi)) - \\beta \\partial_{\\beta} \\varsigma(\\boldsymbol\\pi) \\Bigg]_i (\\partial_{\\alpha} Q)^i_k x^k.
        $$

        and where $\\partial_{\\beta} \\partial_{\\alpha} \mathcal L = \\partial_{\\alpha} \\partial_{\\beta} \mathcal L$ since the second derivatives of $\mathcal L$ are continuous in the neighbourhood of the parameters.

        Arguments:

            action: `ndarray(nactions)`. One-hot action vector
            state: `ndarray(nstates)`. One-hot state vector

        """
        # Obtain the components required for computation of derivatives
        lr             = self.critic.learning_rate
        beta           = self.actor.inverse_softmax_temp
        Qx             = self.critic.Qx(state)
        logits         = beta*Qx
        pu             = fu.softmax(logits)
        du             = action - pu
        dpu_dlogit     = grad.softmax(logits)
        dlogit_dB      = Qx
        dpu_dB         = np.einsum('ij,j->i', dpu_dlogit, dlogit_dB)
        D2Q_lr         = self.critic.hess_Q['learning_rate']
        DQ_lr          = self.critic.dQ['learning_rate']
        DQ_lr_state    = np.dot(DQ_lr, state)
        dpu_dlr        = beta*np.einsum('ij,j->i', dpu_dlogit, DQ_lr_state)
        self.logprob_ += np.dot(action, self.actor.log_prob(Qx))

        # Partial derivative of log probability with respect to inverse softmax temperature
        #  Second order
        self.hess_logprob['inverse_softmax_temp'] += np.dot(action, self.actor.hess_logprob['inverse_softmax_temp'])
        #  First order
        self.d_logprob['inverse_softmax_temp'] += np.dot(action, self.actor.d_logprob['inverse_softmax_temp'])

        # Second and first partial derivative of log probability with respect to learning rate
        #   Second partial derivative with respect to learning rate
        self.hess_logprob['learning_rate'] +=  beta*np.dot(np.einsum('i,ij->j', du, self.critic.hess_Q['learning_rate']) - np.einsum('i,ij->j', dpu_dlr, DQ_lr), state)
        #   First partial derivative with respect to learning rate
        self.d_logprob['learning_rate'] += beta*np.dot(du, DQ_lr_state)

        # Partial derivative with respect to both learning rate and inverse softmax
        self.hess_logprob['learning_rate_inverse_softmax_temp'] += np.dot(du - beta*dpu_dB, DQ_lr_state)

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
            - $\\tilde{\mathbf u}$ be a one-hot vector representing the last trial's action, where at trial 0, $\\tilde{\mathbf u} = \mathbf 0$.
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
        \\partial_{\\alpha} \mathcal L = \\beta \Big[ \\big(\mathbf u - \\varsigma(\\pi)\\big)_i (\\partial_{\\alpha} Q)^i_j x^j \Big],
        $$

        and with respect to $\\beta$

        $$
        \\partial_{\\beta} \mathcal L = u_i \Big(\mathbf I_{n_u \\times n_u} - \\varsigma(\\boldsymbol\\pi)\Big)^i_j Q_{jk} x^k
        $$

        and with respect to $\\rho$

        $$
        \\partial_{\\rho} \mathcal L = u_i \Big(\mathbf I_{n_u \\times n_u} - \\varsigma(\\boldsymbol\\pi)\Big)^i_j \\tilde{u}^j.
        $$

        We also compute the Hessian $\mathbf H$, defined as

        $$
        \mathbf H = \\left[
            \\begin{array}{ccc}
            \\partial^2_{\\alpha} \mathcal L & \\partial_{\\alpha} \\partial_{\\beta} \mathcal L & \\partial_{\\alpha} \\partial_{\\rho} \mathcal L \\\\
            \\partial_{\\beta} \\partial_{\\alpha} \mathcal L & \\partial^2_{\\beta} \mathcal L & \\partial_{\\beta} \\partial_{\\rho} \mathcal L \\\\
            \\partial_{\\rho} \\partial_{\\alpha} \mathcal L & \\partial_{\\rho} \\partial_{\\beta} \mathcal L & \\partial^2_{\\rho} \mathcal L \\\\
            \\end{array}\\right].
        $$

        The components of $\mathbf H$ are virtually identical to that of `RWSoftmaxAgent`, with the exception of the $\\partial_{\\rho} \\partial_{\\alpha} \mathcal L$ and $\\partial_{\\beta} \\partial_{\\rho} \mathcal L$

        $$
        \\partial^2_{\\alpha} \mathcal L = \\beta \Big( (\mathbf u - \\varsigma(\\boldsymbol\\pi))_i (\\partial^2_{\\alpha} \mathbf Q)^i - \\partial_{\\alpha} \\varsigma(\\boldsymbol\\pi)_i (\\partial_{\\alpha} \mathbf Q)^i \Big)_j x^j,
        $$

        $$
        \\partial^2_{\\beta} \mathcal L = u_k \\Bigg(\\frac{(q_i q_i v^i v^i}{z^2} - \\frac{q_i q_i v^i}{z} \\Bigg)^k
        $$

        $$
        \\partial_{\\alpha} \\partial_{\\beta} \mathcal L = \\Bigg[ (u - \\varsigma(\\boldsymbol\\pi)) - \\beta \\partial_{\\beta} \\varsigma(\\boldsymbol\\pi) \\Bigg]_i (\\partial_{\\alpha} Q)^i_k x^k
        $$

        $$
        \\partial_{\\alpha} \\partial_{\\rho} \mathcal L = - \\beta \\Big( \\partial_{\\boldsymbol\\pi} \\varsigma(\\boldsymbol\\pi)_i \\tilde{u}^i \\Big)_j (\\partial_{\\alpha} Q)^j_k x^k
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
        DQ_lr_state    = np.einsum('ij,j->i', DQ_lr, state)
        dpu_dlr        = beta*np.einsum('ij,j->i', dpu_dlogit, DQ_lr_state)

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
        self.d_logprob['learning_rate'] += beta*np.dot(du, DQ_lr_state)

        # Partial derivative with respect to both learning rate and inverse softmax
        self.hess_logprob['learning_rate_inverse_softmax_temp'] += np.dot(du - beta*dpu_dB, DQ_lr_state)

        # Partial derivative with respect to both learning rate and perseveration
        self.hess_logprob['learning_rate_perseveration'] += -beta*np.dot(dpu_drho, DQ_lr_state)

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

class TwoStepStickySoftmaxSARSABellmanMaxAgent(object):
    """ An agent specifically constructed for the two-step task (Daw et al. 2011) including a softmax policy with a SARSA learning rule for the model free value function. The model-based value function backs up the maximal second step action value to the first step, and weights this value with the model-free state-action values.

    This agent includes a pre-computed gradient with respect to the parameters. Second order derivatives have not yet been added.

    Note this agent is not a composition of a `ValueFunction` and `Policy` as the other agents are, on account of some nuances to the two-step task. Further development of `fitr` will split this into the separate component objects.
    """
    def __init__(self,
                 task,
                 learning_rate_1=0.1,
                 learning_rate_2=0.1,
                 inverse_softmax_temp_1=1.,
                 inverse_softmax_temp_2=1.,
                 trace_decay=1.,
                 mb_weight=0.5,
                 perseveration=0.1,
                 rng=np.random.RandomState()):
        """

        Arguments:

            task: `fitr.environments.Graph`
            learning_rate_1: `0 < float < 1`. First step learning rate
            learning_rate_2: `0 < float < 1`. Second step learning rate
            inverse_softmax_temp_1: `float`. First step inverse softmax temperature
            inverse_softmax_temp_2: `float`. Second step inverse softmax temperature
            trace_decay: '0<float<1'. Eligibility trace decay
            mb_weight: '0<float<1'. Proportion of value coming from model based system
            perseveration: `float`. Weight on taking same first-step action as last trial
            rng: `np.random.RandomState`


        """
        self.task = task
        self.nstates = task.nstates
        self.nactions = task.nactions
        self.learning_rates = [learning_rate_1, learning_rate_2]
        self.inverse_softmax_temps = [inverse_softmax_temp_1, inverse_softmax_temp_2]
        self.trace_decay = trace_decay
        self.mb_weight = mb_weight
        self.perseveration = perseveration
        self.rng = rng

        # Set the meta stuff
        self.meta = ['TwoStepStickySoftmaxSARSABellmanMaxAgent']
        self.params = [learning_rate_1, 
                       learning_rate_2, 
                       inverse_softmax_temp_1, 
                       inverse_softmax_temp_2,
                       trace_decay, 
                       mb_weight, 
                       perseveration]

        # Get the transition matrix from the task object and preprocess
        self.T = task.T
        self.T[:,1:,1:] = 0

        # Initialize last-action for sticky softmax
        self.a_last = np.zeros(self.nactions)

        # Initialize value functions and gradients
        self.Q = np.zeros((self.nactions, self.nstates))
        self.Qmb = np.zeros((self.nactions, self.nstates))
        self.Qmf = np.zeros((self.nactions, self.nstates))

        self.dQ = {
            'Qmf': 1-self.mb_weight,
            'Qmb': self.mb_weight,
            'learning_rate_1': np.zeros((self.nactions, self.nstates)),
            'learning_rate_2': np.zeros((self.nactions, self.nstates)),
            'trace_decay': np.zeros((self.nactions, self.nstates)),
            'mb_weight': self.Qmb - self.Qmf
        }

        self.dQmb = {
            'Qmf': np.zeros((self.nactions, self.nstates)),
            'learning_rate_1': np.zeros((self.nactions, self.nstates)),
            'learning_rate_2': np.zeros((self.nactions, self.nstates)),
            'trace_decay': np.zeros((self.nactions, self.nstates))
        }

        self.dQmf = {
            'learning_rate_1': np.zeros((self.nactions, self.nstates)),
            'learning_rate_2': np.zeros((self.nactions, self.nstates)),
            'trace_decay': np.zeros((self.nactions, self.nstates))
        }

        # Initialize derivatives with respect to log-probability
        self.d_logprob = {
            'learning_rate_1': 0,
            'learning_rate_2': 0,
            'inverse_softmax_temp_1': 0,
            'inverse_softmax_temp_2': 0,
            'trace_decay': 0,
            'mb_weight': 0,
            'perseveration': 0
        }

        # Log probability and gradient vector
        self.logprob_ = 0
        self.grad_ = None

    def action_step1(self, state):
        """ Returns the action the agent will take at the first step of a trial.

        This function is separate from the second step action because of the perseveration element.

        Arguments:

            state: `ndarray(nstates)`. First-step state (one-hot vector)

        Returns:

            action: `ndarray(nactions)`. First-step action (one-hot)
        """
        q = np.einsum('ij,j->i', self.Q, state)
        value_logits  = self.inverse_softmax_temps[0]*q
        persev_logits = self.perseveration*self.a_last
        logits = value_logits + persev_logits
        pu = fu.softmax(logits)
        action = self.rng.multinomial(1, pvals=pu)
        self.a_last = action
        return action

    def action_step2(self, state):
        """ Returns the action the agent will take at the second step of a trial.

        This function is separate from the first step action because of the perseveration element in the first step. Also, at the second step, the agent only uses model-free values.

        Arguments:

            state: `ndarray(nstates)`. Second-step state (one-hot vector)

        Returns:

            action: `ndarray(nactions)`. Second-step action (one-hot)
        """
        q = np.einsum('ij,j->i', self.Qmf, state)
        logits  = self.inverse_softmax_temps[1]*q
        pu = fu.softmax(logits)
        action = self.rng.multinomial(1, pvals=pu)
        return action

    def log_prob(self, x, u, x_, u_):
        """ Computes the log-probability of a behavioural sequence on a trial

        Note here we have shortened the variable names since the gradient equations are quite long. This was done in order to facilitate comparison of the gradient equations with the mathematical representations.

        Arguments:

            x: `ndarray(nstates)`. First-step state
            u: `ndarray(nactions)`. First-step action
            x_: `ndarray(nstates)`. Second-step state
            u_: `ndarray(nactions)`. Second-step action


        """
        q          = np.einsum('ij,j->i', self.Q, x)
        q_         = np.einsum('ij,j->i', self.Qmf, x_)
        B1         = self.inverse_softmax_temps[0]
        B2         = self.inverse_softmax_temps[1]
        logits1    = B1*q + self.perseveration*self.a_last
        logits2    = B2*q_
        pu1        = fu.softmax(logits1)
        pu2        = fu.softmax(logits2)
        Dlp_logit1 = np.eye(q.size)  - np.tile(grad.logsumexp(logits1), [q.size, 1])
        Dlp_logit2 = np.eye(q_.size) - np.tile(grad.logsumexp(logits2), [q_.size, 1])
        Dlogit1_q1 = B1
        Dlogit2_q2 = B2
        Dlogit1_B1 = q
        Dlogit2_B2 = q_
        Dlp_q1     = B1*np.eye(q.size)  - np.tile(B1*grad.logsumexp(logits1), [q.size, 1])
        Dlp_q2     = B2*np.eye(q_.size)  - np.tile(B2*grad.logsumexp(logits2), [q_.size, 1])
        Dq1_Q      = x
        Dq2_Q      = x_
        Dq2_Qmf    = x_
        self.dQ['mb_weight'] = self.Qmb - self.Qmf

        # Update log probability
        self.logprob_ += np.dot(u, logits1 - fu.logsumexp(logits1))
        self.logprob_ += np.dot(u_, logits2 - fu.logsumexp(logits2))

        # Compute derivatives
        self.d_logprob['learning_rate_1'] += np.dot(u,  np.einsum('ij,jk,k->i', Dlp_q1, self.dQ['learning_rate_1'], Dq1_Q))
        self.d_logprob['learning_rate_1'] += np.dot(u_, np.einsum('ij,jk,k->i', Dlp_q2, self.dQmf['learning_rate_1'], Dq2_Qmf))
        self.d_logprob['learning_rate_2'] += np.dot(u,  np.einsum('ij,jk,k->i', Dlp_q1, self.dQ['learning_rate_2'], Dq1_Q))
        self.d_logprob['learning_rate_2'] += np.dot(u_, np.einsum('ij,jk,k->i', Dlp_q2, self.dQmf['learning_rate_2'], Dq2_Qmf))
        self.d_logprob['mb_weight'] += np.dot(u,  np.dot(Dlp_q1, np.dot(self.dQ['mb_weight'], Dq1_Q)))

        self.dQ['trace_decay'] = (self.dQ['Qmb']*self.dQmb['Qmf'] + self.dQ['Qmf'])*self.dQmf['trace_decay']
        self.d_logprob['trace_decay'] += np.dot(u,  np.einsum('ij,jk,k->i', Dlp_q1, self.dQ['trace_decay'], Dq1_Q))
        self.d_logprob['trace_decay'] += np.dot(u_, np.einsum('ij,jk,k->i', Dlp_q2, self.dQmf['trace_decay'], Dq2_Qmf))

        self.d_logprob['inverse_softmax_temp_1']  += u@Dlp_logit1@Dlogit1_B1
        self.d_logprob['perseveration']           += u@Dlp_logit1@self.a_last
        self.d_logprob['inverse_softmax_temp_2']  += u_@Dlp_logit2@Dlogit2_B2

        # Update the last action
        self.a_last = u

        # Create gradient vector
        self.grad_ = np.array([self.d_logprob['learning_rate_1'],
                               self.d_logprob['learning_rate_2'],
                               self.d_logprob['inverse_softmax_temp_1'],
                               self.d_logprob['inverse_softmax_temp_2'],
                               self.d_logprob['trace_decay'],
                               self.d_logprob['mb_weight'],
                               self.d_logprob['perseveration']])

    def _log_prob_noderivatives(self, x, u, x_, u_):
        """ Computes the log-probability of an action taken by the agent in a given state without updating partial derivatives with respect to the parameters.

        This function is only implemented for purposes of unit testing for comparison with `autograd` package.

        Arguments:

            x: `ndarray(nstates)`. First-step state
            u: `ndarray(nactions)`. First-step action
            x_: `ndarray(nstates)`. Second-step state
            u_: `ndarray(nactions)`. Second-step action

        """
        q          = np.einsum('ij,j->i', self.Q, x)
        q_         = np.einsum('ij,j->i', self.Qmf, x_)
        logits1    = self.inverse_softmax_temps[0]*q + self.perseveration*self.a_last
        logits2    = self.inverse_softmax_temps[1]*q_

        # Update log probability
        self.logprob_ += np.dot(u, logits1 - fu.logsumexp(logits1))
        self.logprob_ += np.dot(u_, logits2 - fu.logsumexp(logits2))

        self.a_last = u

    def learning(self, x, u, x_, u_, r):
        """ Updates the agent's value functions

        Note here we have shortened the variable names since the gradient equations are quite long. This was done in order to facilitate comparison of the gradient equations with the mathematical representations.

        Arguments:

            x: `ndarray(nstates)`. First-step state
            u: `ndarray(nactions)`. First-step action
            x_: `ndarray(nstates)`. Second-step state
            u_: `ndarray(nactions)`. Second-step action
            r: `float`. Reward received in the trial

        """
        # Eligibility trace
        z1   = np.outer(u, x)
        z2   = np.outer(u_, x_) + self.trace_decay*z1

        # Precompute some reused values
        u_Qmfx_     = u_.T@self.Qmf@x_
        uQmfx       = u.T@self.Qmf@x
        u_DQmflr1x_ = u_.T@self.dQmf['learning_rate_1']@x_
        uDQmflr1x   = u.T@self.dQmf['learning_rate_1']@x

        u_DQmflr2x_ = u_.T@self.dQmf['learning_rate_2']@x_
        uDQmflr2x   = u.T@self.dQmf['learning_rate_2']@x

        u_DQmftdx_ = u_.T@self.dQmf['trace_decay']@x_
        uDQmftdx   = u.T@self.dQmf['trace_decay']@x

        # First partial derivatives of model-free values with respect to parameters
        self.dQmf['learning_rate_1'] += (u_Qmfx_- uQmfx)*z1 + self.learning_rates[0]*((u_DQmflr1x_ - uDQmflr1x)*z1 - u_DQmflr1x_*z2)
        self.dQmf['learning_rate_2'] += self.learning_rates[0]*(u_DQmflr2x_ - uDQmflr2x)*z1 + (r - u_Qmfx_ - self.learning_rates[1]*u_DQmflr2x_)*z2
        self.dQmf['trace_decay']  += self.learning_rates[0]*(u_DQmftdx_ - uDQmftdx)*z1 + self.learning_rates[1]*((r - u_Qmfx_)*z1 - u_DQmftdx_*z2)

        # Update model-free value function
        self.Qmf += self.learning_rates[0]*u_Qmfx_*z1 - self.learning_rates[0]*uQmfx*z1 + self.learning_rates[1]*r*z2 - self.learning_rates[1]*u_Qmfx_*z2

        # First partial derivatives with respect to model-based value and net value function
        DmaxQmf_Qmf = grad.matrix_max(self.Qmf, axis=0)
        DQmb_maxQmf = np.tile(np.sum(np.sum(self.T, axis=0), axis=1), [self.nactions, 1])
        self.dQmb['Qmf'] = DQmb_maxQmf*DmaxQmf_Qmf
        self.dQ['Qmf']   = 1 - self.mb_weight + self.mb_weight*self.dQmb['Qmf']

        DmaxQmf_lr1 = np.einsum('ij,ij->j', DmaxQmf_Qmf, self.dQmf['learning_rate_1'])
        self.dQmb['learning_rate_1'] = np.einsum('ijk,j->ik', self.T, DmaxQmf_lr1)
        self.dQ['learning_rate_1'] = self.mb_weight*self.dQmb['learning_rate_1'] + (1-self.mb_weight)*self.dQmf['learning_rate_1']

        DmaxQmf_lr2 = np.einsum('ij,ij->j', DmaxQmf_Qmf, self.dQmf['learning_rate_2'])
        self.dQmb['learning_rate_2'] = np.einsum('ijk,j->ik', self.T, DmaxQmf_lr2)
        self.dQ['learning_rate_2'] = self.mb_weight*self.dQmb['learning_rate_2'] + (1-self.mb_weight)*self.dQmf['learning_rate_2']
        self.dQ['trace_decay'] = self.dQ['Qmf']*self.dQmf['trace_decay']

        # Update the model based value function and weight it with the model free one
        maxQmf = np.max(self.Qmf, axis=0)
        self.Qmb    = np.einsum('ijk,j->ik', self.T, maxQmf)
        self.Q      = self.mb_weight*self.Qmb + (1-self.mb_weight)*self.Qmf


    def _learning_noderivatives(self, x, u, x_, u_, r):
        """ Updates the agent's value functions without computing derivatives.

        This method exists only to facilitate unit testing with comparison against autograd.

        Arguments:

            x: `ndarray(nstates)`. First-step state
            u: `ndarray(nactions)`. First-step action
            x_: `ndarray(nstates)`. Second-step state
            u_: `ndarray(nactions)`. Second-step action
            r: `float`. Reward received in the trial

        """
        # Eligibility trace
        z1   = np.outer(u, x)
        z2   = np.outer(u_, x_) + self.trace_decay*z1

        u_Qmfx_     = u_.T@self.Qmf@x_
        uQmfx       = u.T@self.Qmf@x

        # Update model-free value function
        self.Qmf += self.learning_rates[0]*u_Qmfx_*z1 - self.learning_rates[0]*uQmfx*z1 + self.learning_rates[1]*r*z2 - self.learning_rates[1]*u_Qmfx_*z2

        # Update the model based value function and weight it with the model free one
        maxQmf = np.max(self.Qmf, axis=0)
        self.Qmb    = np.einsum('ijk,j->ik', self.T, maxQmf)
        self.Q      = self.mb_weight*self.Qmb + (1-self.mb_weight)*self.Qmf


    def generate_data(self, ntrials, state_only=False):
        """ For the parent agent, this function generates data from a Markov Decision Process (MDP) task

        Arguments:

            ntrials: `int` number of trials
            state_only: `bool`. If the eligibility trace is only an `nstate` dimensional vector (i.e. for a Pavlovian conditioning model) then set to `True`. For instumental models, the eligibility trace should be an `nactions` by `nstates` matrix, so keep this to `False` in that case.

        Returns:

            `fitr.data.BehaviouralData`
        """
        data = BehaviouralData(1)
        data.add_subject(0, self.params, self.meta)
        for t in range(ntrials):
            state1  = self.task.observation()
            action1 = self.action_step1(state1)
            state2, _, _ = self.task.step(action1)
            action2 = self.action_step2(state2)
            _, reward, _ = self.task.step(action2)
            data.update(0, np.hstack((state1, action1, state2, action2, reward)))
            self.learning(state1, action1, state2, action2, reward)
        data.make_tensor_representations()
        return data
