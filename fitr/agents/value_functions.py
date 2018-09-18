import autograd.numpy as np
import fitr.gradients as grad

class ValueFunction(object):
    """ A general value function object.

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

        env: A `fitr.environments.Graph`
    """
    def __init__(self, env):
        self.nstates  = env.nstates
        self.nactions = env.nactions
        self.V = np.zeros(self.nstates)
        self.Q = np.zeros((self.nactions, self.nstates))
        self.rpe = [0]
        self.etrace = None
        self.dV = {}
        self.dQ = {}

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

    def grad_Qx(self, x):
        """ Compute gradient of action values for a given state

        $$
        \mathcal Q(\mathbf x, :) = \mathbf Q \mathbf x,
        $$

        where the gradient is defined as

        $$
        \\frac{\\partial}{\\partial \mathbf Q} \mathcal Q(\mathbf x, :) = \mathbf 1 \mathbf x^\\top,
        $$

        Arguments:

            x: `ndarray((nstates,))` one-hot state vector

        Returns:

            `ndarray((nactions,))` vector of values for actions in the given state
        """
        return np.tile(x, [self.nactions, 1])

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

    def grad_uQx(self, u, x):
        """ Compute derivative of value of taking action $\mathbf u$ in state $\mathbf x$ with respect to value function parameters $\mathbf Q$

        $$
        \mathcal Q(\mathbf x, \mathbf u) = \mathbf u^\\top \mathbf Q \mathbf x,
        $$

        where the derivative is defined as

        $$
        \\frac{\\partial}{\\partial \mathbf Q} \mathcal Q(\mathbf x, \mathbf u) = \mathbf u \mathbf x^\\top,
        $$

        Arguments:

            u: `ndarray((nactions,))` one-hot action vector
            x: `ndarray((nstates,))` one-hot state vector

        Returns:

            Scalar value of action $\mathbf u$ in state $\mathbf x$
        """
        return np.einsum('i,j->ij', u, x)

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

    def grad_Vx(self, x):
        """ Compute the gradient of state value function with respect to parameters $\mathbf v$

        $$
        \mathcal V(\mathbf x) = \mathbf v^\\top \mathbf x,
        $$

        where the gradient is defined as

        $$
        \\nabla_{\mathbf v} \mathcal V(\mathbf x) = \mathbf x
        $$

        Arguments:

            x: `ndarray((nstates,))` one-hot state vector

        Returns:

            Scalar value of state $\mathbf x$
        """
        return x

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

        # Store gradient of learning rule with respect to learning rate
        self.dQ = {
            'learning_rate': np.zeros(self.Q.shape),
            'rpe': np.zeros(self.Q.shape)}
        self.hess_Q = {'learning_rate': np.zeros(self.Q.shape)}

    def update(self, x, u, r, x_, u_):
        """ Computes the value function update of the instrumental Rescorla-Wagner learning rule and computes derivative with respect to the learning rate.

        This derivative is defined as

        $$
        \\frac{\\partial}{\\partial \\alpha} \mathcal Q(\mathbf x, \mathbf u; \\alpha) = \\delta \mathbf u \mathbf x^\\top + \\frac{\\partial}{\\partial \\alpha} \mathcal Q(\mathbf x, \mathbf u; \\alpha) (1-\\alpha \mathbf u \mathbf x^\\top)
        $$

        and the second order derivative with respect to learning rate is

        $$
        \\frac{\\partial}{\\partial \\alpha} \mathcal Q(\mathbf x, \mathbf u; \\alpha) = - 2 \\mathbf u \\mathbf x^\\top \\partial_\\alpha \mathcal Q(\mathbf x, \mathbf u; \\alpha) + \\partial^2_\\alpha \mathcal Q(\mathbf x, \mathbf u; \\alpha) (1 - \\alpha \mathbf u \mathbf x^\\top)
        $$

        Arguments:

            x: `ndarray((nstates, ))`. State vector
            u: `ndarray((nactions, ))`. Action vector
            r: `float`. Reward received
            x_: `ndarray((nstates, ))`. For compatibility
            u_: `ndarray((nactions, ))`. For compatibility

        """
        rpe    = r - self.uQx(u, x)
        self.rpe.append(rpe)
        z = np.outer(u, x)
        rpe_z = rpe*z # Compute this ahead of time to avoid repeated operations
        lr_z = self.learning_rate*z
        self.hess_Q['learning_rate'] = -2*z*self.dQ['learning_rate'] + self.hess_Q['learning_rate']*(1 - lr_z)
        self.dQ['learning_rate'] = rpe_z + self.dQ['learning_rate']*(1 - lr_z)
        self.Q += self.learning_rate*rpe_z

    def _update_noderivatives(self, x, u, r, x_, u_):
        """ Computes the value function update of the instrumental Rescorla-Wagner learning rule without the derivative.

        This function is identical to `.update()` method except without the derivative computations. It is implemented solely for the purpose of unit testing the gradient calculations against `autograd`.

        """
        rpe    = r - self.uQx(u, x)
        z = np.outer(u, x)
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
        self.etrace = np.zeros(self.Q.shape)
        self.dQ = {
            'learning_rate': np.zeros(self.Q.shape),
            'discount_factor': np.zeros(self.Q.shape),
            'trace_decay': np.zeros(self.Q.shape)
        }
        self.d_etrace = {
            'trace_decay': np.zeros(self.Q.shape),
            'discount_factor': np.zeros(self.Q.shape)
        }

    def update(self, x, u, r, x_, u_):
        """ Computes value function updates and their derivatives for the Q-learning model """

        # ELIGIBILITY TRACE
        # Reset derivatives if eligibility trace was reset at start of trial
        if np.all(np.equal(self.etrace, 0)):
            self.d_etrace['discount_factor'] = np.zeros(self.etrace.shape)
            self.d_etrace['trace_decay'] = np.zeros(self.etrace.shape)

        # Compute derivatives
        self.d_etrace['discount_factor'] = self.trace_decay*(self.etrace + self.discount_factor*self.d_etrace['discount_factor'])
        self.d_etrace['trace_decay'] = self.discount_factor*(self.etrace + self.trace_decay*self.d_etrace['trace_decay'])

        # Update trace
        self.etrace = np.einsum('a,s->as', u, x) + self.discount_factor*self.trace_decay*self.etrace

        # REWARD PREDICTION ERROR
        # Compute derivatives
        dmaxQx_ = grad.max(self.Qx(x_))
        d_rpe_Q = self.discount_factor*np.outer(dmaxQx_, x_) - np.outer(u, x)
        d_rpe_learningrate = np.sum(self.dQ['learning_rate']*d_rpe_Q)
        d_rpe_discount = np.sum(self.dQ['discount_factor']*d_rpe_Q) + self.Qmax(x_)
        d_rpe_tracedecay = np.sum(self.dQ['trace_decay']*d_rpe_Q)

        # Compute RPE
        rpe = r + self.discount_factor*self.Qmax(x_) - self.uQx(u, x)
        self.rpe.append(rpe)

        # Q PARAMETERS
        # Compute derivatives
        self.dQ['learning_rate'] += (rpe + self.learning_rate*d_rpe_learningrate)*self.etrace
        self.dQ['discount_factor'] += self.learning_rate*(d_rpe_discount*self.etrace + rpe*self.d_etrace['discount_factor'])
        self.dQ['trace_decay'] += self.learning_rate*(d_rpe_tracedecay*self.etrace + rpe*self.d_etrace['trace_decay'])

        # Update value function
        self.Q += self.learning_rate*rpe*self.etrace

    def _update_noderivatives(self, x, u, r, x_, u_):
        """ Computes value function updates without computing derivatives.

        This function is identical to `.update()` method except without the derivative computations. It is implemented solely for the purpose of unit testing the gradient calculations against `autograd`.
        """
        self.etrace = np.einsum('a,s->as', u, x) + self.discount_factor*self.trace_decay*self.etrace
        rpe = r + self.discount_factor*self.Qmax(x_) - self.uQx(u, x)
        self.rpe.append(rpe)
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

        self.d_rpe = {
            'Q': np.zeros(self.Q.shape),
            'learning_rate': 0,
            'discount_factor': 0,
            'trace_decay': 0
        }

        self.dQ = {
            'learning_rate': np.zeros(self.Q.shape),
            'discount_factor': np.zeros(self.Q.shape),
            'trace_decay': np.zeros(self.Q.shape)
        }

        self.d_etrace = {
            'trace_decay': np.zeros(self.Q.shape),
            'discount_factor': np.zeros(self.Q.shape)
        }

        self.hess_Q = {
            'learning_rate': np.zeros(self.Q.shape),
            'discount_factor': np.zeros(self.Q.shape),
            'trace_decay': np.zeros(self.Q.shape)
        }

        self.hess_rpe = {
            'learning_rate': 0,
            'discount_factor': 0,
            'trace_decay': 0
        }

        self.d_etrace = {
            'trace_decay': np.zeros(self.Q.shape),
            'discount_factor': np.zeros(self.Q.shape)
        }

    def update(self, x, u, r, x_, u_):
        """ Computes value function updates and their derivatives for the SARSA model """
        # Precompute some repeatedly used values
        z     = np.outer(u,x)
        z_    = np.outer(u_, x_)
        uQx   = self.uQx(u, x)
        u_Qx_ = self.uQx(u_, x_)

        # ELIGIBILITY TRACE
        # Reset derivatives if eligibility trace was reset at start of trial
        if np.all(np.equal(self.etrace, 0)):
            self.hess_etrace['discount_factor'] = np.zeros(self.etrace.shape)
            self.hess_etrace['trace_decay'] = np.zeros(self.etrace.shape)
            self.d_etrace['discount_factor'] = np.zeros(self.etrace.shape)
            self.d_etrace['trace_decay'] = np.zeros(self.etrace.shape)

        # Compute derivatives
        self.hess_etrace['discount_factor'] = self.trace_decay*(2*self.d_etrace['discount_factor'] + self.discount_factor*self.hess_etrace['discount_factor'])
        self.hess_etrace['trace_decay'] = self.discount_factor*(2*self.d_etrace['trace_decay'] + self.trace_decay*self.hess_etrace['trace_decay'])
        self.d_etrace['discount_factor'] = self.trace_decay*(self.etrace + self.discount_factor*self.d_etrace['discount_factor'])
        self.d_etrace['trace_decay'] = self.discount_factor*(self.etrace + self.trace_decay*self.d_etrace['trace_decay'])

        # Update trace
        self.etrace = z + self.discount_factor*self.trace_decay*self.etrace

        # REWARD PREDICTION ERROR
        # Compute derivatives
        #   Second order
        self.hess_rpe['learning_rate']   = np.sum(self.hess_Q['learning_rate']*self.d_rpe['Q'])
        self.hess_rpe['discount_factor'] = np.sum(self.hess_Q['discount_factor']*self.d_rpe['Q']) + np.einsum('i,ij,j->', u_, self.d_Q['discount_factor'], x_)
        self.hess_rpe['trace_decay']     = np.sum(self.hess_Q['trace_decay']*self.d_rpe['Q'])

        #   First order
        self.d_rpe['Q'] = self.discount_factor*z_ - z
        self.d_rpe['learning_rate'] = np.sum(self.dQ['learning_rate']*self.d_rpe['Q'])
        self.d_rpe['discount_factor'] = np.sum(self.dQ['discount_factor']*self.d_rpe['Q']) + u_Qx_
        self.d_rpe['trace_decay'] = np.sum(self.dQ['trace_decay']*self.d_rpe['Q'])

        # Compute RPE
        rpe = r + self.discount_factor*u_Qx_ - uQx
        self.rpe.append(rpe)

        # Q PARAMETERS
        # Compute derivatives
        #   Second order
        self.hess_Q['learning_rate'] += (2*self.d_rpe['learning_rate'] + self.learning_rate*self.hess_rpe['learning_rate'])*self.etrace
        self.hess_Q['discount_factor'] += self.learning_rate*(self.hess_rpe['discount_factor']*self.etrace + rpe*self.d_etrace['discount_factor'] + self.d_rpe['discount_factor']*self.d_etrace['discount_factor'] + rpe*self.hess_etrace['discount_factor'])
        self.hess_Q['trace_decay'] += self.learning_rate*(self.hess_rpe['trace_decay']*self.etrace + rpe*self.d_etrace['trace_decay'] + self.d_rpe['trace_decay']*self.d_etrace['trace_decay'] + rpe*self.hess_etrace['trace_decay'])

        #   First order
        self.dQ['learning_rate'] += (rpe + self.learning_rate*self.d_rpe['learning_rate'])*self.etrace
        self.dQ['discount_factor'] += self.learning_rate*(self.d_rpe['discount_factor']*self.etrace + rpe*self.d_etrace['discount_factor'])
        self.dQ['trace_decay'] += self.learning_rate*(self.d_rpe['trace_decay']*self.etrace + rpe*self.d_etrace['trace_decay'])

        # Update value function
        self.Q += self.learning_rate*rpe*self.etrace

    def _update_noderivatives(self, x, u, r, x_, u_):
        """ Computes value function updates without computing derivatives.

        This function is identical to `.update()` method except without the derivative computations. It is implemented solely for the purpose of unit testing the gradient calculations against `autograd`.
        """
        self.etrace = np.einsum('a,s->as', u, x) + self.discount_factor*self.trace_decay*self.etrace
        rpe = r + self.discount_factor*self.uQx(u_, x_) - self.uQx(u, x)
        self.rpe.append(rpe)
        self.Q += self.learning_rate*rpe*self.etrace
