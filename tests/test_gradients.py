import autograd.numpy as np
from autograd import grad as gradient
from autograd import elementwise_grad, jacobian
from fitr import utils
from fitr import gradients as grad
from fitr.environments import TwoArmedBandit
from fitr.environments import TwoStep
from fitr.agents import RWSoftmaxAgent
from fitr.agents import RWStickySoftmaxAgent
from fitr.agents.policies import SoftmaxPolicy
from fitr.agents.policies import StickySoftmaxPolicy
from fitr.agents.value_functions import ValueFunction
from fitr.agents.value_functions import InstrumentalRescorlaWagnerLearner
from fitr.agents.value_functions import QLearner
from fitr.agents.value_functions import SARSALearner

def test_logsumexp():
    x = np.array([1., 0., 0.])
    eps = np.ones(x.size)*1e-8
    grad_fitr     = grad.logsumexp(x)
    grad_autograd = gradient(utils.logsumexp)(x)
    err = np.linalg.norm(grad_fitr-grad_autograd)
    assert(err < 1e-5)

def test_max():
    rng = np.random.RandomState(236)
    ag_max = jacobian(np.max)
    for i in range(20):
        x = rng.normal(size=5)**2
        ag_grad = ag_max(x)
        fitr_grad = grad.max(x)
        assert(np.linalg.norm(ag_grad-fitr_grad) < 1e-6)

def test_softmaxpolicy_gradients():
    x = np.array([0., 1., 0., 0.])
    B = 1.

    smpol = SoftmaxPolicy(B)
    smpol.log_prob(x)

    ag_x = jacobian(SoftmaxPolicy(1.)._log_prob_noderivatives)(x)
    assert(np.linalg.norm(ag_x-smpol.d_logprob['action_values']) < 1e-5)

    fB = lambda B: SoftmaxPolicy(B)._log_prob_noderivatives(x)
    ag_B = jacobian(fB)(B)
    assert(np.linalg.norm(ag_B-smpol.d_logprob['inverse_softmax_temp']) < 1e-5)

def test_stickysoftmaxpolicy_gradients():
    x = np.array([0., 1., 0., 0.])
    u = np.array([0., 0., 1., 0.])
    B = 1.
    p = 0.01
    policy = StickySoftmaxPolicy(B, p)
    policy.a_last = u

    def fB(B):
        policy = StickySoftmaxPolicy(B, p)
        policy.a_last = u
        return policy._log_prob_noderivatives(x)

    def fp(p):
        policy = StickySoftmaxPolicy(B, p)
        policy.a_last = u
        return policy._log_prob_noderivatives(x)

    def fx(x):
        policy = StickySoftmaxPolicy(B, p)
        policy.a_last = u
        return policy._log_prob_noderivatives(x)

    policy.log_prob(x)
    fitr_gxB = policy.d_logprob['inverse_softmax_temp']
    fitr_gxp = policy.d_logprob['perseveration']
    fitr_gxx = policy.d_logprob['action_values']

    ag_gxB = jacobian(fB)(B)
    ag_gxp = jacobian(fp)(p)
    ag_gxx = jacobian(fx)(x)

    assert(np.linalg.norm(ag_gxB-fitr_gxB) < 1e-5)
    assert(np.linalg.norm(ag_gxp-fitr_gxp) < 1e-5)
    assert(np.linalg.norm(ag_gxx-fitr_gxx) < 1e-5)

def test_grad_Qx():
    x = np.array([1., 0., 0.])
    task = TwoArmedBandit()
    v = ValueFunction(task)
    v.Q = np.array([[1., 2., 3.], [4., 5., 6.]])
    def vfx(Q):
        v.Q = Q
        return v.Qx(x)

    agQx = elementwise_grad(vfx)(v.Q)
    gQ = v.grad_Qx(x)
    assert(np.linalg.norm(agQx-gQ) < 1e-5)

def test_grad_uQx():
    x = np.array([1., 0., 0.])
    u = np.array([0., 1.])
    task = TwoArmedBandit()
    v = ValueFunction(task)
    v.Q = np.array([[1., 2., 3.], [4., 5., 6.]])
    def vfx(Q):
        v.Q = Q
        return v.uQx(u, x)

    agQx = elementwise_grad(vfx)(v.Q)
    gQ = v.grad_uQx(u, x)
    assert(np.linalg.norm(agQx-gQ) < 1e-5)

def test_grad_Vx():
    x = np.array([1., 0., 0.])
    task = TwoArmedBandit()
    v = ValueFunction(task)
    v.V = np.array([1., 2., 3.])
    def vfx(V):
        v.V = V
        return v.Vx(x)

    agVx = elementwise_grad(vfx)(v.V)
    gV = v.grad_Vx(x)
    assert(np.linalg.norm(agVx-gV) < 1e-5)

def test_grad_instrumantalrwupdate():
    lr = 0.1
    task = TwoArmedBandit()
    q = InstrumentalRescorlaWagnerLearner(task, learning_rate=lr)

    x  = np.array([1., 0., 0.])
    u1  = np.array([1., 0.])
    u2  = np.array([0., 1.])
    x_1 = np.array([0., 1., 0.])
    x_2 = np.array([0., 0., 1.])
    r1 = 1.0
    r2 = 0.0

    q.update(x, u1, r1, x_1, None)
    q.update(x, u2, r2, x_2, None)
    q.update(x, u2, r1, x_1, None)
    q.update(x, u1, r2, x_2, None)
    q.update(x, u1, r1, x_1, None)
    fitr_grad = q.dQ['learning_rate']

    def fq(lr):
        m = InstrumentalRescorlaWagnerLearner(task, learning_rate=lr)
        m._update_noderivatives(x, u1, r1, x_1, None)
        m._update_noderivatives(x, u2, r2, x_2, None)
        m._update_noderivatives(x, u2, r1, x_1, None)
        m._update_noderivatives(x, u1, r2, x_2, None)
        m._update_noderivatives(x, u1, r1, x_1, None)
        return m.Q
    agQ = jacobian(fq)(lr)

    assert(np.linalg.norm(fitr_grad-agQ) < 1e-6)

def test_grad_qlearnerupdate():
    ntrials = 7
    def make_mdp_trials():
        rng = np.random.RandomState(3256)
        X1 = np.tile(np.array([1., 0., 0., 0., 0.]), [ntrials, 1])
        X2 = rng.multinomial(1, pvals=[0., 0.5, 0.5, 0., 0.], size=ntrials)
        U1 = rng.multinomial(1, pvals=[0.5, 0.5], size=ntrials)
        U2 = rng.multinomial(1, pvals=[0.5, 0.5], size=ntrials)
        X3 = rng.multinomial(1, pvals=[0., 0., 0., 0.5, 0.5], size=ntrials)
        R  = np.array([0., 0., 0., 1., 0.])
        return X1, X2, U1, U2, X3, R

    # GRADIENTS WITH FITR
    X1, X2, U1, U2, X3, R = make_mdp_trials()
    q = QLearner(TwoStep(), learning_rate=0.1, discount_factor=0.9, trace_decay=0.95)
    for i in range(ntrials):
        q.etrace = np.zeros(q.Q.shape)
        x = X1[i]; u = U1[i]; x_= X2[i]; r = R@x_
        q.update(x, u, r, x_, None)
        u_ = U2[i]; x = x_; u = u_; x_ = X3[i]; r  = R@x_
        q.update(x, u, r, x_, None)

    # AUTOGRAD
    def agf_lr(lr):
        X1, X2, U1, U2, X3, R = make_mdp_trials()
        q = QLearner(TwoStep(), learning_rate=lr, discount_factor=0.9, trace_decay=0.95)
        for i in range(ntrials):
            q.etrace = np.zeros((2, 5))
            x = X1[i]; u = U1[i]; x_= X2[i]; r = R@x_
            q._update_noderivatives(x, u, r, x_, None)
            u_ = U2[i]; x = x_; u = u_; x_ = X3[i]; r = R@x_
            q._update_noderivatives(x, u, r, x_, None)
        return q.Q

    def agf_dc(dc):
        X1, X2, U1, U2, X3, R = make_mdp_trials()
        q = QLearner(TwoStep(), learning_rate=0.1, discount_factor=dc, trace_decay=0.95)
        for i in range(ntrials):
            q.etrace = np.zeros((2, 5))
            x = X1[i]; u = U1[i]; x_= X2[i]; r = R@x_
            q._update_noderivatives(x, u, r, x_, None)
            u_ = U2[i]; x = x_; u = u_; x_ = X3[i]; r = R@x_
            q._update_noderivatives(x, u, r, x_, None)
        return q.Q

    def agf_et(et):
        X1, X2, U1, U2, X3, R = make_mdp_trials()
        q = QLearner(TwoStep(), learning_rate=0.1, discount_factor=0.9, trace_decay=et)
        for i in range(ntrials):
            q.etrace = np.zeros((2, 5))
            x = X1[i]; u = U1[i]; x_= X2[i]; r = R@x_
            q._update_noderivatives(x, u, r, x_, None)
            u_ = U2[i]; x = x_; u = u_; x_ = X3[i]; r = R@x_
            q._update_noderivatives(x, u, r, x_, None)
        return q.Q

    # Ensure all are producing same value functions
    qlist = [agf_lr(0.1), agf_dc(0.9), agf_et(0.95), q.Q]
    assert(np.all(np.stack(np.all(np.equal(a, b)) for a in qlist for b in qlist)))

    # Check partial derivative of Q with respect to learning rate
    assert(np.linalg.norm(q.dQ['learning_rate']-jacobian(agf_lr)(0.1)) < 1e-6)

    # Check partial derivative of Q with respect to discount factor
    assert(np.linalg.norm(q.dQ['discount_factor']-jacobian(agf_dc)(0.9)) < 1e-6)

    # Check partial derivative of Q with respect to trace decay
    assert(np.linalg.norm(q.dQ['trace_decay']-jacobian(agf_et)(0.95)) < 1e-6)


def test_grad_sarsalearnerupdate():
    ntrials = 7
    def make_mdp_trials():
        rng = np.random.RandomState(3256)
        X1 = np.tile(np.array([1., 0., 0., 0., 0.]), [ntrials, 1])
        X2 = rng.multinomial(1, pvals=[0., 0.5, 0.5, 0., 0.], size=ntrials)
        U1 = rng.multinomial(1, pvals=[0.5, 0.5], size=ntrials)
        U2 = rng.multinomial(1, pvals=[0.5, 0.5], size=ntrials)
        X3 = rng.multinomial(1, pvals=[0., 0., 0., 0.5, 0.5], size=ntrials)
        U3 = rng.multinomial(1, pvals=[0.5, 0.5], size=ntrials)
        R  = np.array([0., 0., 0., 1., 0.])
        return X1, X2, U1, U2, X3, U3, R

    # GRADIENTS WITH FITR
    X1, X2, U1, U2, X3, U3, R = make_mdp_trials()
    q = SARSALearner(TwoStep(), learning_rate=0.1, discount_factor=0.9, trace_decay=0.95)
    for i in range(ntrials):
        q.etrace = np.zeros(q.Q.shape)
        x = X1[i]; u = U1[i]; x_= X2[i]; r = R@x_; u_ = U2[i];
        q.update(x, u, r, x_, u_)
        x = x_; u = u_; x_ = X3[i]; u_ = U3[i]; r  = R@x_
        q.update(x, u, r, x_, u_)

    # AUTOGRAD
    def agf_lr(lr):
        X1, X2, U1, U2, X3, U3, R = make_mdp_trials()
        q = SARSALearner(TwoStep(), learning_rate=lr, discount_factor=0.9, trace_decay=0.95)
        for i in range(ntrials):
            q.etrace = np.zeros(q.Q.shape)
            x = X1[i]; u = U1[i]; x_= X2[i]; r = R@x_; u_ = U2[i];
            q._update_noderivatives(x, u, r, x_, u_)
            x = x_; u = u_; x_ = X3[i];  u_ = U3[i]; r  = R@x_
            q._update_noderivatives(x, u, r, x_, u_)
        return q.Q

    def agf_dc(dc):
        X1, X2, U1, U2, X3, U3, R = make_mdp_trials()
        q = SARSALearner(TwoStep(), learning_rate=0.1, discount_factor=dc, trace_decay=0.95)
        for i in range(ntrials):
            q.etrace = np.zeros(q.Q.shape)
            x = X1[i]; u = U1[i]; x_= X2[i]; r = R@x_; u_ = U2[i];
            q._update_noderivatives(x, u, r, x_, u_)
            x = x_; u = u_; x_ = X3[i];  u_ = U3[i]; r  = R@x_
            q._update_noderivatives(x, u, r, x_, u_)
        return q.Q

    def agf_et(et):
        X1, X2, U1, U2, X3, U3, R = make_mdp_trials()
        q = SARSALearner(TwoStep(), learning_rate=0.1, discount_factor=0.9, trace_decay=et)
        for i in range(ntrials):
            q.etrace = np.zeros(q.Q.shape)
            x = X1[i]; u = U1[i]; x_= X2[i]; r = R@x_; u_ = U2[i];
            q._update_noderivatives(x, u, r, x_, u_)
            x = x_; u = u_; x_ = X3[i];  u_ = U3[i]; r  = R@x_
            q._update_noderivatives(x, u, r, x_, u_)
        return q.Q

    # Ensure all are producing same value functions
    qlist = [agf_lr(0.1), agf_dc(0.9), agf_et(0.95), q.Q]
    assert(np.all(np.stack(np.all(np.equal(a, b)) for a in qlist for b in qlist)))

    # Check partial derivative of Q with respect to learning rate
    assert(np.linalg.norm(q.dQ['learning_rate']-jacobian(agf_lr)(0.1)) < 1e-6)

    # Check partial derivative of Q with respect to discount factor
    assert(np.linalg.norm(q.dQ['discount_factor']-jacobian(agf_dc)(0.9)) < 1e-6)

    # Check partial derivative of Q with respect to trace decay
    assert(np.linalg.norm(q.dQ['trace_decay']-jacobian(agf_et)(0.95)) < 1e-6)


def test_rwsoftmaxagent():
    lr = 0.1
    B  = 1.5
    task = TwoArmedBandit()
    q = RWSoftmaxAgent(task, learning_rate=lr, inverse_softmax_temp=B)

    x  = np.array([1., 0., 0.])
    u1  = np.array([1., 0.])
    u2  = np.array([0., 1.])
    x_1 = np.array([0., 1., 0.])
    x_2 = np.array([0., 0., 1.])
    r1 = 1.0
    r2 = 0.0

    q.log_prob(x, u1)
    q.learning(x, u1, r1, x_1, None)
    q.log_prob(x, u2)
    q.learning(x, u2, r2, x_2, None)
    q.log_prob(x, u2)
    q.learning(x, u2, r1, x_1, None)
    q.log_prob(x, u1)
    q.learning(x, u1, r2, x_2, None)
    q.log_prob(x, u1)
    q.learning(x, u1, r1, x_1, None)

    fitr_lrgrad = q.d_logprob['learning_rate']
    fitr_istgrad= q.d_logprob['inverse_softmax_temp']


    def fq(lr):
        m = RWSoftmaxAgent(task, learning_rate=lr, inverse_softmax_temp=1.5)
        m._log_prob_noderivatives(x, u1)
        m.critic._update_noderivatives(x, u1, r1, x_1, None)
        m._log_prob_noderivatives(x, u2)
        m.critic._update_noderivatives(x, u2, r2, x_2, None)
        m._log_prob_noderivatives(x, u2)
        m.critic._update_noderivatives(x, u2, r1, x_1, None)
        m._log_prob_noderivatives(x, u1)
        m.critic._update_noderivatives(x, u1, r2, x_2, None)
        m._log_prob_noderivatives(x, u1)
        m.critic._update_noderivatives(x, u1, r1, x_1, None)
        return m.logprob_

    def fB(beta):
        m = RWSoftmaxAgent(task, learning_rate=0.1, inverse_softmax_temp=beta)
        m._log_prob_noderivatives(x, u1)
        m.critic._update_noderivatives(x, u1, r1, x_1, None)
        m._log_prob_noderivatives(x, u2)
        m.critic._update_noderivatives(x, u2, r2, x_2, None)
        m._log_prob_noderivatives(x, u2)
        m.critic._update_noderivatives(x, u2, r1, x_1, None)
        m._log_prob_noderivatives(x, u1)
        m.critic._update_noderivatives(x, u1, r2, x_2, None)
        m._log_prob_noderivatives(x, u1)
        m.critic._update_noderivatives(x, u1, r1, x_1, None)
        return m.logprob_

    agQ = jacobian(fq)(lr)
    agB = jacobian(fB)(B)

    assert(np.linalg.norm(fitr_lrgrad-agQ) < 1e-6)
    assert(np.linalg.norm(fitr_istgrad-agB) < 1e-6)

def test_rwstickysoftmaxagent():
    lr = 0.1
    B  = 1.5
    p  = 0.01
    task = TwoArmedBandit()
    q = RWStickySoftmaxAgent(task,
                             learning_rate=lr,
                             inverse_softmax_temp=B,
                             perseveration=p)

    x  = np.array([1., 0., 0.])
    u1  = np.array([1., 0.])
    u2  = np.array([0., 1.])
    x_1 = np.array([0., 1., 0.])
    x_2 = np.array([0., 0., 1.])
    r1 = 1.0
    r2 = 0.0

    q.log_prob(x, u1)
    q.learning(x, u1, r1, x_1, None)
    q.log_prob(x, u2)
    q.learning(x, u2, r2, x_2, None)
    q.log_prob(x, u2)
    q.learning(x, u2, r1, x_1, None)
    q.log_prob(x, u1)
    q.learning(x, u1, r2, x_2, None)
    q.log_prob(x, u1)
    q.learning(x, u1, r1, x_1, None)

    fitr_lrgrad = q.d_logprob['learning_rate']
    fitr_istgrad= q.d_logprob['inverse_softmax_temp']
    fitr_pgrad  = q.d_logprob['perseveration']

    def fq(lr):
        m = RWStickySoftmaxAgent(task,
                                 learning_rate=lr,
                                 inverse_softmax_temp=1.5,
                                 perseveration=0.01)
        m._log_prob_noderivatives(x, u1)
        m.critic._update_noderivatives(x, u1, r1, x_1, None)
        m._log_prob_noderivatives(x, u2)
        m.critic._update_noderivatives(x, u2, r2, x_2, None)
        m._log_prob_noderivatives(x, u2)
        m.critic._update_noderivatives(x, u2, r1, x_1, None)
        m._log_prob_noderivatives(x, u1)
        m.critic._update_noderivatives(x, u1, r2, x_2, None)
        m._log_prob_noderivatives(x, u1)
        m.critic._update_noderivatives(x, u1, r1, x_1, None)
        return m.logprob_

    def fB(beta):
        m = RWStickySoftmaxAgent(task,
                                 learning_rate=0.1,
                                 inverse_softmax_temp=beta,
                                 perseveration=0.01)
        m._log_prob_noderivatives(x, u1)
        m.critic._update_noderivatives(x, u1, r1, x_1, None)
        m._log_prob_noderivatives(x, u2)
        m.critic._update_noderivatives(x, u2, r2, x_2, None)
        m._log_prob_noderivatives(x, u2)
        m.critic._update_noderivatives(x, u2, r1, x_1, None)
        m._log_prob_noderivatives(x, u1)
        m.critic._update_noderivatives(x, u1, r2, x_2, None)
        m._log_prob_noderivatives(x, u1)
        m.critic._update_noderivatives(x, u1, r1, x_1, None)
        return m.logprob_

    def fp(p):
        m = RWStickySoftmaxAgent(task,
                                 learning_rate=0.1,
                                 inverse_softmax_temp=1.5,
                                 perseveration=p)
        m._log_prob_noderivatives(x, u1)
        m.critic._update_noderivatives(x, u1, r1, x_1, None)
        m._log_prob_noderivatives(x, u2)
        m.critic._update_noderivatives(x, u2, r2, x_2, None)
        m._log_prob_noderivatives(x, u2)
        m.critic._update_noderivatives(x, u2, r1, x_1, None)
        m._log_prob_noderivatives(x, u1)
        m.critic._update_noderivatives(x, u1, r2, x_2, None)
        m._log_prob_noderivatives(x, u1)
        m.critic._update_noderivatives(x, u1, r1, x_1, None)
        return m.logprob_

    agQ = jacobian(fq)(lr)
    agB = jacobian(fB)(B)
    agp = jacobian(fp)(p)

    assert(np.linalg.norm(fitr_lrgrad-agQ) < 1e-6)
    assert(np.linalg.norm(fitr_istgrad-agB) < 1e-6)
    assert(np.linalg.norm(fitr_pgrad-agp) < 1e-6)
