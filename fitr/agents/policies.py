import autograd.numpy as np
import fitr.utils as fu
import fitr.gradients as grad
import fitr.hessians as hess

class SoftmaxPolicy(object):
    """ Action selection by sampling from a multinomial whose parameters are given by a softmax.

    Action sampling is

    $$
    \mathbf u \sim \mathrm{Multinomial}(1, \mathbf p=\\varsigma(\mathbf v)).
    $$

    Parameters of that distribution are

    $$
    p(\mathbf u|\mathbf v) = \\varsigma(\mathbf v) = \\frac{e^{\\beta \mathbf v}}{\sum_{i}^{|\mathbf v|} e^{\\beta v_i}}.
    $$

    Arguments:

        inverse_softmax_temp: Inverse softmax temperature $\\beta$
        rng: `np.random.RandomState` object

    """
    def __init__(self, inverse_softmax_temp=1., rng=np.random.RandomState()):
        self.inverse_softmax_temp = inverse_softmax_temp
        self.rng  = rng

        # Storage for first order partial derivatives
        self.d_logprob = {
            'logits': None,
            'inverse_softmax_temp': None,
            'action_values': None
        }

        # Storage for second order partial derivatives
        self.hess_logprob = {
            'inverse_softmax_temp': None,
            'action_values': None
        }

    def log_prob(self, x):
        """ Computes the log-probability of an action $\mathbf u$, in addition to computing derivatives up to second order

        $$
        \log p(\mathbf u|\mathbf v) = \\beta \mathbf v - \log \sum_{v_i} e^{\\beta \mathbf v_i}
        $$

        Arguments:

            x: State vector of type `ndarray((nstates,))`

        Returns:

            Scalar log-probability
        """
        # Compute logits
        Bx  = self.inverse_softmax_temp*x

        # Hessians
        HB, Hx = hess.log_softmax(self.inverse_softmax_temp, x)
        self.hess_logprob['inverse_softmax_temp'] = HB
        self.hess_logprob['action_values'] = Hx

        # Derivatives
        #  Grad LSE wrt Logits
        Dlse = grad.logsumexp(Bx)

        # Grad logprob wrt logits
        self.d_logprob['logits'] = np.eye(x.size) - Dlse

        #  Grad logprob wrt inverse softmax temp
        self.d_logprob['inverse_softmax_temp'] = np.dot(self.d_logprob['logits'], x)

        # Grad logprob wrt action values `x`
        B = np.eye(x.size)*self.inverse_softmax_temp
        Dlsetile = np.tile(self.inverse_softmax_temp*Dlse, [x.size, 1])
        self.d_logprob['action_values'] = B - Dlsetile

        # Compute log-probability of actions
        LSE = fu.logsumexp(Bx)
        if not np.isfinite(LSE): LSE = 0.
        return Bx - LSE

    def _log_prob_noderivatives(self, x):
        """ Computes the log-probability of an action $\mathbf u$ without computing derivatives.

        This is here only to facilitate unit testing of the `.log_prob` method by comparison against `autograd`.
        """
        # Compute logits
        Bx  = self.inverse_softmax_temp*x

        # Compute log-probability of actions
        LSE = fu.logsumexp(Bx)
        if not np.isfinite(LSE): LSE = 0.
        return Bx - LSE

    def action_prob(self, x):
        """ Computes the softmax """
        return fu.softmax(self.inverse_softmax_temp*x)

    def sample(self, x):
        """ Samples from the action distribution """
        return self.rng.multinomial(1, pvals=self.action_prob(x))

class StickySoftmaxPolicy(object):
    """ Action selection by sampling from a multinomial whose parameters are given by a softmax, but with accounting for the tendency to perseverate (i.e. choosing the previously used action without considering its value).

    Let $\mathbf u_{t-1} = (u_{t-1}^{(i)})_{i=1}^{|\mathcal U|}$ be a one hot vector representing the action taken at the last step, and $\\beta^\\rho$ be an inverse softmax temperature for the influence of this last action.

    Action sampling is thus:

    $$
    \mathbf u \sim \mathrm{Multinomial}(1, \mathbf p=\\varsigma(\mathbf v, \mathbf u_{t-1})).
    $$

    Parameters of that distribution are

    $$
    p(\mathbf u|\mathbf v, \mathbf u_{t-1}) = \\varsigma(\mathbf v, \mathbf u_{t-1}) = \\frac{e^{\\beta \mathbf v + \\beta^\\rho \mathbf u_{t-1}}}{\sum_{i}^{|\mathbf v|} e^{\\beta v_i + \\beta^\\rho u_{t-1}^{(i)}}}.
    $$

    Arguments:

        inverse_softmax_temp: Inverse softmax temperature $\\beta$
        perseveration: Inverse softmax temperature $\\beta^\\rho$ capturing the tendency to repeat the last action taken.
        rng: `np.random.RandomState` object

    """
    def __init__(self, inverse_softmax_temp=1., perseveration=0.01, rng=np.random.RandomState()):
        self.inverse_softmax_temp = inverse_softmax_temp
        self.perseveration        = perseveration
        self.rng  = rng
        self.a_last = [0]

        # Storage for first order partial derivatives
        self.d_logprob = {
            'inverse_softmax_temp': None,
            'perseveration': None,
            'action_values': None
        }

        # Storage for second order partial derivatives
        self.hess_logprob = {
            'inverse_softmax_temp': None,
            'perseveration': None,
            'action_values': None
        }

    def log_prob(self, x):
        """ Computes the log-probability of an action $\mathbf u$

        $$
        \log p(\mathbf u|\mathbf v, \mathbf u_{t-1}) = \\big(\\beta \mathbf v + \\beta^\\rho \mathbf u_{t-1}) - \log \sum_{v_i} e^{\\beta \mathbf v_i + \\beta^\\rho u_{t-1}^{(i)}}
        $$

        Arguments:

            x: State vector of type `ndarray((nactions,))`

        Returns:

            Scalar log-probability
        """
        # Compute logits
        Bx  = self.inverse_softmax_temp*x
        stickiness = self.perseveration*self.a_last
        logits = Bx + stickiness

        # Hessians
        HB, Hp, Hx, _ = hess.log_stickysoftmax(B, p, x, self.a_last)
        self.hess_logprob['inverse_softmax_temp'] = HB
        self.hess_logprob['perseveration'] = Hp
        self.hess_logprob['action_values'] = Hx

        # Derivatives
        #  Grad LSE wrt Logits
        Dlse = grad.logsumexp(logits)

        #  Partial derivative with respect to inverse softmax temp
        self.d_logprob['inverse_softmax_temp'] = x - np.dot(Dlse, x)
        self.d_logprob['perseveration'] = self.a_last - np.dot(Dlse, self.a_last)

        # Gradient with respect to x
        B = np.eye(x.size)*self.inverse_softmax_temp
        Dlsetile = np.tile(self.inverse_softmax_temp*Dlse, [x.size, 1])
        self.d_logprob['action_values'] = B - Dlsetile

        LSE = fu.logsumexp(logits)
        if not np.isfinite(LSE): LSE = 0.
        return logits - LSE

    def _log_prob_noderivatives(self, x):
        """ Computes the log-probability of an action $\mathbf u$ without computing derivatives.

        This is here only to facilitate unit testing of the `.log_prob` method by comparison against `autograd`.
        """
        # Compute logits
        Bx  = self.inverse_softmax_temp*x
        stickiness = self.perseveration*self.a_last
        logits = Bx + stickiness
        LSE = fu.logsumexp(logits)
        if not np.isfinite(LSE): LSE = 0.
        return logits - LSE

    def action_prob(self, x):
        """ Computes the softmax

        Arguments:

            x: `ndarray((nactions,))` one-hot state vector

        Returns:

            `ndarray((nactions,))` vector of action probabilities
        """
        stickiness = self.perseveration*self.a_last
        return fu.softmax(self.inverse_softmax_temp*x + stickiness)

    def sample(self, x):
        """ Samples from the action distribution

        Arguments:

            x: `ndarray((nactions,))` one-hot state vector

        Returns:

            `ndarray((nactions,))` one-hot action vector
        """
        a_new = self.rng.multinomial(1, pvals=self.action_prob(x))
        self.a_last = a_new
        return a_new

class EpsilonGreedyPolicy(object):
    """ A policy that takes the maximally valued action with probability $1-\\epsilon$, otherwise chooses randomlyself.

    Arguments:

        epsilon: Probability of not taking the action with highest value
        rng: `numpy.random.RandomState` object
    """
    def __init__(self, epsilon=0.1, rng=np.random.RandomState()):
        self.epsilon = epsilon
        self.rng  = rng

    def action_prob(self, x):
        """ Creates vector of action probabilities for e-greedy policy

        Arguments:

            x: `ndarray((nstates,))` one-hot state vector

        Returns:

            `ndarray((nstates,))` vector of action probabilities
        """
        p = np.zeros(x.size)
        p[np.argmax(x)] = 1 - self.epsilon
        p[p == 0.] = np.epsilon/(x.size-1)
        return p

    def sample(self, x):
        """ Samples from the action distribution

        Arguments:

            x: `ndarray((nstates,))` one-hot state vector

        Returns:

            `ndarray((nstates,))` one-hot action vector
        """
        return self.rng.multinomial(1, pvals=self.action_prob(x))
