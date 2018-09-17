import autograd.numpy as np
from autograd import grad as gradient
from autograd import elementwise_grad, jacobian, hessian
from fitr import utils
from fitr import gradients as grad
from fitr import hessians as hess
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
    grad_fitr = grad.logsumexp(x)
    hess_fitr = hess.logsumexp(x)
    grad_autograd = gradient(utils.logsumexp)(x)
    hess_autograd = hessian(utils.logsumexp)(x)
    grad_err = np.linalg.norm(grad_fitr-grad_autograd)
    hess_err = np.linalg.norm(hess_fitr-hess_autograd)
    assert(grad_err < 1e-6)
    assert(hess_err < 1e-6)

def test_max():
    rng = np.random.RandomState(236)
    ag_max = jacobian(np.max)
    for i in range(20):
        x = rng.normal(size=5)**2
        ag_grad = ag_max(x)
        fitr_grad = grad.max(x)
        assert(np.linalg.norm(ag_grad-fitr_grad) < 1e-6)

def test_sigmoid():
    x = np.linspace(-5, 5, 10)
    f = lambda x: utils.sigmoid(x)
    ag = elementwise_grad(f)(x)
    fg = grad.sigmoid(x)
    assert(np.all(np.linalg.norm(ag-fg) < 1e-6))

def test_softmax():
    x = np.arange(5)+1
    x = x.astype(np.float)
    f = lambda x: utils.softmax(x)
    gx = jacobian(f)
    agx = gx(x)
    fitrgx = grad.softmax(x)
    assert(np.linalg.norm(agx-fitrgx) < 1e-6)

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

def test_softmax_policy_hessian():
    q = np.arange(5)+1
    q = q.astype(np.float)
    q = q-np.max(q)
    B = 1.5

    fB = lambda B: B*q - utils.logsumexp(B*q)
    fq = lambda q: B*q - utils.logsumexp(B*q)

    aghB = hessian(fB)
    aghq = hessian(fq)

    autograd_hessB = aghB(B)
    autograd_hessq = aghq(q)

    HB, Hq = hess.log_softmax(B, q)
    assert(np.linalg.norm(autograd_hessB - HB) < 1e-6)
    assert(np.linalg.norm(autograd_hessq - Hq) < 1e-6)

def test_sticky_softmax_policy_hessian():
    q = np.arange(5)+1
    q = q.astype(np.float)
    q = q-np.max(q)
    u = np.array([0., 0., 1., 0., 0.])
    u = u - np.max(u)
    B = 1.5
    p = 0.01

    f = lambda x: x[0]*q + x[1]*u - utils.logsumexp(x[0]*q + x[1]*u)
    fq = lambda q: B*q + p*u - utils.logsumexp(B*q + p*u)
    fu = lambda u: B*q + p*u - utils.logsumexp(B*q + p*u)

    aghq = hessian(fq)
    aghu = hessian(fu)

    autograd_hessq = aghq(q)
    autograd_hessu = aghu(u)
    agH = hessian(f)(np.array([B, p]))[0]

    HB, Hp, HBp, Hq, Hu = hess.log_stickysoftmax(B, p, q, u)
    fitrH = np.array([[HB[0], HBp[0]],[HBp[0], Hp[0]]])
    assert(np.linalg.norm(agH - fitrH) < 1e-6)
    assert(np.linalg.norm(autograd_hessq - Hq) < 1e-6)
    assert(np.linalg.norm(autograd_hessu - Hu) < 1e-6)

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
    fitr_hess = q.hess_Q['learning_rate']

    def fq(lr):
        m = InstrumentalRescorlaWagnerLearner(task, learning_rate=lr)
        m._update_noderivatives(x, u1, r1, x_1, None)
        m._update_noderivatives(x, u2, r2, x_2, None)
        m._update_noderivatives(x, u2, r1, x_1, None)
        m._update_noderivatives(x, u1, r2, x_2, None)
        m._update_noderivatives(x, u1, r1, x_1, None)
        return m.Q

    agQ = jacobian(fq)(lr)
    ahQ = hessian(fq)(lr)

    assert(np.linalg.norm(fitr_grad-agQ) < 1e-6)
    assert(np.linalg.norm(fitr_hess-ahQ) < 1e-6)

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
    q = RWSoftmaxAgent(task,
                       learning_rate=lr,
                       inverse_softmax_temp=B)

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

    fitr_grad = q.grad_
    fitr_hess = q.hess_

    def f(w):
        m = RWSoftmaxAgent(task, learning_rate=w[0], inverse_softmax_temp=w[1])
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

    agJ = jacobian(f)(np.array([lr, B]))
    agH = hessian(f)(np.array([lr, B]))

    assert(np.linalg.norm(agJ - q.grad_))
    assert(np.linalg.norm(agH - q.hess_))

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

    def f(w):
        m = RWStickySoftmaxAgent(task,
                                 learning_rate=w[0],
                                 inverse_softmax_temp=w[1],
                                 perseveration=w[2])
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

    w = np.array([lr, B, p])
    ag = jacobian(f)(w)
    aH = hessian(f)(w)

    assert(np.linalg.norm(q.grad_ - ag) < 1e-6)
    assert(np.linalg.norm(q.hess_ - aH) < 1e-6)
