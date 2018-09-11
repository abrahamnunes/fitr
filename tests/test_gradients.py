import autograd.numpy as np
from autograd import grad as gradient
from autograd import elementwise_grad, jacobian
from fitr import utils
from fitr import gradients as grad
from fitr.environments import TwoArmedBandit
from fitr.agents.policies import SoftmaxPolicy
from fitr.agents.policies import StickySoftmaxPolicy
from fitr.agents.value_functions import ValueFunction
from fitr.agents.value_functions import InstrumentalRescorlaWagnerLearner

def test_logsumexp():
    x = np.array([1., 0., 0.])
    eps = np.ones(x.size)*1e-8
    grad_fitr     = grad.logsumexp(x)
    grad_autograd = gradient(utils.logsumexp)(x)
    err = np.linalg.norm(grad_fitr-grad_autograd)
    assert(err < 1e-5)

def test_softmaxpolicy_gradients():
    x = np.array([0., 1., 0., 0.])
    B = 1.

    gx = lambda x: SoftmaxPolicy(B).grad_log_prob(x)['x']
    gB = lambda B: SoftmaxPolicy(B).grad_log_prob(x)['inverse_softmax_temp']

    ag_x = jacobian(SoftmaxPolicy(1.).log_prob)(x)
    assert(np.linalg.norm(ag_x-gx(x)) < 1e-5)

    fB = lambda B: SoftmaxPolicy(B).log_prob(x)
    ag_B = jacobian(fB)(B)
    assert(np.linalg.norm(ag_B-gB(B)) < 1e-5)

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
        return policy.log_prob(x)

    def fp(p):
        policy = StickySoftmaxPolicy(B, p)
        policy.a_last = u
        return policy.log_prob(x)

    def fx(x):
        policy = StickySoftmaxPolicy(B, p)
        policy.a_last = u
        return policy.log_prob(x)

    fitr_grads = policy.grad_log_prob(x)
    fitr_gxB = fitr_grads['inverse_softmax_temp']
    fitr_gxp = fitr_grads['perseveration']
    fitr_gxx = fitr_grads['x']

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

    fitr_grad = q.grad_update(x, u1, r1, x_1, None)
    q.update(x, u1, r1, x_1, None)
    fitr_grad = q.grad_update(x, u2, r2, x_2, None)
    q.update(x, u2, r2, x_2, None)
    fitr_grad = q.grad_update(x, u2, r1, x_1, None)
    q.update(x, u2, r1, x_1, None)
    fitr_grad = q.grad_update(x, u1, r2, x_2, None)
    q.update(x, u1, r2, x_2, None)
    fitr_grad = q.grad_update(x, u1, r1, x_1, None)
    q.update(x, u1, r1, x_1, None)

    def fq(lr):
        m = InstrumentalRescorlaWagnerLearner(task, learning_rate=lr)
        m.update(x, u1, r1, x_1, None)
        m.update(x, u2, r2, x_2, None)
        m.update(x, u2, r1, x_1, None)
        m.update(x, u1, r2, x_2, None)
        m.update(x, u1, r1, x_1, None)
        return m.Q
    agQ = jacobian(fq)(lr)
    assert(np.linalg.norm(fitr_grad-agQ) < 1e-6)
