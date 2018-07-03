import numpy as np

class ValueFunction(object):
    """ A general value function object.

    A value function here is task specific and consists of several attributes:

    - `nstates`: The number of states in the task, $|\mathcal X|$
    - `nactions`: Number of actions in the task, $|\mathcal U|$
    - `V`: State value function $\mathbf v = \mathcal V(\mathbf x)$
    - `Q`: State-action value function $\mathbf Q = \mathcal Q(\mathbf x, \mathbf u)$
    - `etrace`: An eligibility trace (optional)

    Note that in general we rely on matrix-vector notation for value functions, rather than function notation. Vectors in the mathematical typesetting are by default column vectors.

    Arguments:

        env: A `fitr.environments.Graph`
    """
    def __init__(self, env):
        self.nstates  = env.nstates
        self.nactions = env.nactions
        self.V = np.zeros(self.nstates)
        self.Q = np.zeros((self.nactions, self.nstates))
        self.etrace = None

    def Qx(self, x):
        """ Compute action values for a given state

        $$
        \mathcal Q(\mathbf x, :) = \mathbf Q \mathbf x
        $$

        Arguments:

            x: `ndarray((nstates,))` one-hot state vector

        Returns:

            `ndarray((nactions,))` vector of values for actions in the given state
        """
        return np.einsum('as,s->a', self.Q, x)

    def Qmax(self, x):
        """ Return maximal action value for given state

        $$
        \max_{u_i}\mathcal Q(\mathbf x, u_i) = \max_{\mathbf u'} \mathbf u'^\\top \mathbf Q \mathbf x
        $$

        Arguments:

            x: `ndarray((nstates,))` one-hot state vector

        Returns:

            Scalar value of the maximal action value at the given state
        """
        return np.max(np.einsum('as,s->a', self.Q, x))

    def Qmean(self, x):
        """ Return mean action value for given state

        $$
        Mean \\big(\mathcal Q(\mathbf x, :)\\big) = \\frac{1}{|\mathcal U|} \mathbf 1^\\top \mathbf Q \mathbf x
        $$

        Arguments:

            x: `ndarray((nstates,))` one-hot state vector

        Returns:

            Scalar value of the maximal action value at the given state
        """
        return np.mean(np.einsum('as,s->a', self.Q, x))

    def uQx(self, u, x):
        """ Compute value of taking action $\mathbf u$ in state $\mathbf x$

        $$
        \mathcal Q(\mathbf x, \mathbf u) = \mathbf u^\\top \mathbf Q \mathbf x
        $$

        Arguments:

            u: `ndarray((nactions,))` one-hot action vector
            x: `ndarray((nstates,))` one-hot state vector

        Returns:

            Scalar value of action $\mathbf u$ in state $\mathbf x$
        """
        return np.einsum('a,as,s->', u, self.Q, x)

    def Vx(self, x):
        """ Compute value of state $\mathbf x$

        $$
        \mathcal V(\mathbf x) = \mathbf v^\\top \mathbf x
        $$

        Arguments:

            x: `ndarray((nstates,))` one-hot state vector

        Returns:

            Scalar value of state $\mathbf x$
        """
        return np.einsum('s,s->', self.V, x)

    def update(self, x, u, r, x_, u_):
        """ Updates the value function

        In the context of the base `ValueFunction` class, this is merely a placeholder. The specific update rule will depend on the specific value function desired.

        Arguments:

            x: `ndarray((nstates,))` one-hot state vector
            u: `ndarray((nactions,))` one-hot action vector
            r: Scalar reward
            x_: `ndarray((nstates,))` one-hot next-state vector
            u_: `ndarray((nactions,))` one-hot next-action vector
        """
        pass

class DummyLearner(ValueFunction):
    """ A critic/value function for the random learner

    This class actually contributes nothing except identifying that a value function has been chosen for an `Agent` object

    Arguments:

        env: A `fitr.environments.Graph`
    """
    def __init__(self, env, **kwargs):
        super().__init__(env)

class InstrumentalRescorlaWagnerLearner(ValueFunction):
    """ Learns an instrumental control policy through one-step error-driven updates of the state-action value function

    The instrumental Rescorla-Wagner rule is as follows:

    $$
    \mathbf Q \\gets \mathbf Q + \\alpha \\big(r - \mathbf u^\\top \mathbf Q \mathbf x \\big) \mathbf u \mathbf x^\\top,
    $$

    where $0 < \\alpha < 1$ is the learning rate, and where the reward prediction error (RPE) is $\\delta = (r - \mathbf u^\\top \mathbf Q \mathbf x)$.

    $$

    Arguments:

        env: A `fitr.environments.Graph`
        learning_rate: Learning rate $\\alpha$

    """
    def __init__(self, env, learning_rate=0.1):
        self.learning_rate = learning_rate
        super().__init__(env)

    def update(self, x, u, r, x_, u_):
        rpe    = r - self.uQx(u, x)
        self.Q += self.learning_rate*rpe*np.einsum('a,s->as', u, x)

class QLearner(ValueFunction):
    """ Learns an instrumental control policy through Q-learning

    The Q-learning rule is as follows:

    $$
    \mathbf Q \\gets \mathbf Q + \\alpha \\big(r + \\gamma \max_{\mathbf u'} \mathbf u'^\\top \mathbf Q \mathbf x' - \mathbf u^\\top \mathbf Q \mathbf x \\big) \mathbf z,
    $$

    where $0 < \\alpha < 1$ is the learning rate, $0 \leq \\gamma \leq 1$ is a discount factor, and where the reward prediction error (RPE) is $\\delta = (r + \\gamma \max_{\mathbf u'} \mathbf u'^\\top \mathbf Q \mathbf x' - \mathbf u^\\top \mathbf Q \mathbf x)$. We have also included an eligibility trace $\mathbf z$ defined as

    $$
    \mathbf z = \mathbf u \mathbf x^\\top +  \\gamma \\lambda \mathbf z
    $$

    Arguments:

        env: A `fitr.environments.Graph`
        learning_rate: Learning rate $\\alpha$
        discount_factor: Discount factor $\\gamma$
        trace_decay: Eligibility trace decay $\\lambda$

    """
    def __init__(self, env,
                 learning_rate=0.1,
                 discount_factor=1.,
                 trace_decay=1.):
        self.learning_rate   = learning_rate
        self.discount_factor = discount_factor
        self.trace_decay     = trace_decay
        super().__init__(env)

    def update(self, x, u, r, x_, u_):
        target = r + self.discount_factor*self.Qmax(x_)
        self.etrace = np.einsum('a,s->as', u, x) + self.discount_factor*self.trace_decay*self.etrace
        rpe    = target - self.uQx(u, x)
        self.Q += self.learning_rate*rpe*self.etrace

class SARSALearner(ValueFunction):
    """ Learns an instrumental control policy through the SARSA learning rule

    The SARSA learning rule is as follows:

    $$
    \mathbf Q \\gets \mathbf Q + \\alpha \\big(r + \\gamma \mathbf u'^\\top \mathbf Q \mathbf x' - \mathbf u^\\top \mathbf Q \mathbf x \\big) \mathbf z,
    $$

    where $0 < \\alpha < 1$ is the learning rate, $0 \leq \\gamma \leq 1$ is a discount factor, and where the reward prediction error (RPE) is $\\delta = (r + \\gamma \mathbf u'^\\top \mathbf Q \mathbf x' - \mathbf u^\\top \mathbf Q \mathbf x)$. We have also included an eligibility trace $\mathbf z$ defined as

    $$
    \mathbf z = \mathbf u \mathbf x^\\top +  \\gamma \\lambda \mathbf z
    $$

    Arguments:

        env: A `fitr.environments.Graph`
        learning_rate: Learning rate $\\alpha$
        discount_factor: Discount factor $\\gamma$
        trace_decay: Eligibility trace decay $\\lambda$

    """
    def __init__(self, env,
                 learning_rate=0.1,
                 discount_factor=1.,
                 trace_decay=1.):
        self.learning_rate   = learning_rate
        self.discount_factor = discount_factor
        self.trace_decay     = trace_decay
        super().__init__(env)

    def update(self, x, u, r, x_, u_):
        target = r + self.discount_factor*self.uQx(u_, x_)
        self.etrace = np.einsum('a,s->as', u, x) + self.discount_factor*self.trace_decay*self.etrace
        rpe    = target - self.uQx(u, x)
        self.Q += self.learning_rate*rpe*self.etrace
