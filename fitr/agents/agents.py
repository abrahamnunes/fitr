import numpy as np
from fitr.agents.policies import *
from fitr.agents.value_functions import *
from fitr.data import BehaviouralData

class Agent(object):
    """
    Base class for synthetic RL agents

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

class BanditAgent(Agent):
    """ A base class for agents in bandit tasks (i.e. with one step).

    This mainly has implications for generating data
    """
    def __init__(self, task):
        super().__init__(task)

    def generate_data(self, ntrials):
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

    This mainly has implications for generating data
    """
    def __init__(self, task):
        super().__init__(task)

    def generate_data(self, ntrials):
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
    """
    An agent that simply selects random actions at each trial
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
    """ An agent that uses the SARSA learning rule and a softmax policy """
    def __init__(self,
                 task,
                 lr=np.random.uniform(0.1, 0.7),
                 dc=np.random.uniform(0.8, 1.0),
                 et=np.random.uniform(0.8, 1.0),
                 cr=np.random.uniform(2, 7)):
        super().__init__(task)
        self.meta = ['SARSASoftmax']
        self.params = [lr, dc, et, cr]
        self.actor  = SoftmaxPolicy(inverse_softmax_temp = cr)
        self.critic = SARSALearner(task,
                                   learning_rate=lr,
                                   discount_factor=dc,
                                   trace_decay=et)

    def action(self, state):
        return self.actor.sample(self.critic.Qx(state))

    def learning(self, state, action, reward, next_state, next_action):
        self.critic.update(state, action, reward, next_state, next_action)

class QLearningSoftmaxAgent(MDPAgent):
    """ An agent that uses the Q-learning rule and a softmax policy """
    def __init__(self,
                 task,
                 lr=np.random.uniform(0.1, 0.7),
                 dc=np.random.uniform(0.8, 1.0),
                 et=np.random.uniform(0.8, 1.0),
                 cr=np.random.uniform(2, 7)):
        super().__init__(task)
        self.meta = ['QSoftmax']
        self.params = [lr, dc, et, cr]
        self.actor  = SoftmaxPolicy(inverse_softmax_temp = cr)
        self.critic = QLearner(task,
                               learning_rate=lr,
                               discount_factor=dc,
                               trace_decay=et)

    def action(self, state):
        return self.actor.sample(self.critic.Qx(state))

    def learning(self, state, action, reward, next_state, next_action):
        self.critic.update(state, action, reward, next_state, next_action)

class RWSoftmaxAgent(BanditAgent):
    def __init__(self, task,
                 lr = np.random.uniform(0.1, 0.7),
                 cr = np.random.uniform(0.01, 2)):
        super().__init__(task)
        self.meta = ['RWSoftmaxAgent']
        lr  = lr
        cr  = cr
        self.params = [lr, cr]
        self.actor  = SoftmaxPolicy(inverse_softmax_temp = cr)
        self.critic = InstrumentalRescorlaWagnerLearner(task, learning_rate=lr)

    def action(self, state):
        return self.actor.sample(self.critic.Qx(state))

    def learning(self, state, action, reward, next_state, next_action):
        self.critic.update(state, action, reward, next_state, None)

class RWSoftmaxAgentRewardSensitivity(BanditAgent):
    def __init__(self, task,
                 lr = np.random.uniform(0.1, 0.7),
                 cr = np.random.uniform(0.01, 2),
                 rs = np.random.uniform(0.1, 0.3)):
        super().__init__(task)
        self.meta = ['RWSoftmaxAgent']
        self.r = rs
        self.params = [lr, cr, self.r]
        self.actor  = SoftmaxPolicy(inverse_softmax_temp = cr)
        self.critic = InstrumentalRescorlaWagnerLearner(task, learning_rate=lr)

    def action(self, state):
        return self.actor.sample(self.critic.Qx(state))

    def learning(self, state, action, reward, next_state, next_action):
        reward *= self.r
        self.critic.update(state, action, reward, next_state, None)
