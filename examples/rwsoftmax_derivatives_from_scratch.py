from autograd import jacobian, hessian, elementwise_grad
import autograd.numpy as np
import fitr.utils as fu
import fitr.gradients as grad
import fitr.hessians as hess
from fitr.environments import TwoArmedBandit
from fitr.agents import RWSoftmaxAgent

lr = 0.1
B = 1.5
ntrials = 20
task = TwoArmedBandit(rng = np.random.RandomState(234))
q = RWSoftmaxAgent(task,
                   learning_rate=lr,
                   inverse_softmax_temp=B,
                   rng = np.random.RandomState(523))

# Generate data
X = []
U = []
R = []
X_ = []
for t in range(ntrials):
    x = task.observation();         X.append(x)
    u = q.action(x);                U.append(u)
    x_, r, _ = task.step(u);        X_.append(x_); R.append(r)
    q.learning(x, u, r, x_, None)


# AUTOGRAD
def reverse(w):
    Q = np.zeros((task.nactions, task.nstates))
    L = 0
    for t in range(ntrials):
        x = X[t]; u = U[t]; r = R[t]; x_ = X_[t]
        z      = np.einsum('i,j->ij', u, x)
        qx     = np.einsum('ij,j->i', Q, x)
        qxu    = np.einsum('i,ij,j->', u, Q, x)
        logits = w[1]*qx
        logits = logits - np.max(logits)
        L += np.dot(u, logits - fu.logsumexp(logits))
        Q += w[0]*(r - qxu)*z
    return L, Q

w = np.array([lr, B])

f = lambda w: reverse(w)[0]
fQ = lambda w: reverse(w)[1]
ag = jacobian(f)(w)
ah = hessian(f)(w)
agQ = elementwise_grad(fQ)(w)
ahQ = hessian(fQ)(w)


# MANUAL
Q  = np.zeros((task.nactions, task.nstates))
dQ_dlr = np.zeros(Q.shape)
d2Q_dlr2 = np.zeros(Q.shape)
dlp_dlr = 0
d2lp_dlr2 = 0
d2lp_dB2 = 0
dlp_dB  = 0
d2lp_dlrdB = 0


AGENT = RWSoftmaxAgent(task, learning_rate=lr, inverse_softmax_temp=B)

I  = np.eye(task.nactions)
BI = B*I

L = 0
rpe = 0
for t in range(ntrials):
    x = X[t]; u = U[t]; r = R[t]; x_ = X_[t]
    z      = np.einsum('i,j->ij', u, x)
    qx     = np.einsum('ij,j->i', Q, x)
    logits = B*qx
    logits = logits - np.max(logits)

    # LOG PROBABILITY
    AGENT.log_prob(x, u)
    L += np.dot(u, logits - fu.logsumexp(logits))

    # Second derivatives
    #   Components
    pu         = fu.softmax(logits)
    du         = u - pu
    dpu_dlogit = grad.softmax(logits)
    dlogit_dB  = qx
    dpu_dB     = np.einsum('ij,j->i', dpu_dlogit, dlogit_dB)
    dpu_dlr    = B*np.einsum('ij,jk,k->i', dpu_dlogit, dQ_dlr, x)

    #   With respect to learning rate
    d2lp_dlr2 += B*np.einsum('k,k->', np.einsum('i,ij->j', du, d2Q_dlr2) - np.einsum('i,ij->j', dpu_dlr, dQ_dlr), x)

    # With respect to learning rate followed by softmax
    #d2lp_dlrdB += np.einsum('i,ij,j', du, dQ_dlr, x) - B*np.einsum('i,ij,j->', dpu_dB, dQ_dlr, x)
    d2lp_dlrdB += np.einsum('i,ij,j->', u - pu - B*dpu_dB, dQ_dlr, x)

    #   With respect to inverse softmax
    D2lp_dB2, D2lp_dqx2 = hess.log_softmax(B, qx)
    d2lp_dB2 += np.dot(u, D2lp_dB2)

    # Compose the hessian
    H = np.array([[d2lp_dlr2, d2lp_dlrdB], [d2lp_dlrdB, d2lp_dB2]])

    # First derivatives for log probability
    #   Components
    dlp_dlogit = I - np.tile(fu.softmax(logits), [task.nactions, 1])
    dlp_dlr +=  B*np.einsum('i,ij,j', du, dQ_dlr, x) #  B*((u-pu).T@dQ_dlr@x) # - pu.T@dQ_dlr@x)  #np.dot(u, np.dot(dlp_dq, dq_dlr))
    dlp_dB += np.einsum('i,ij,jk,k->', u, dlp_dlogit, Q, x) #np.dot(u, np.dot(dlp_dlogit, dlogit_dB))

    # LEARNING
    # Second derivatives
    d2Q_dlr2 = -2*z*dQ_dlr + d2Q_dlr2*(1 - lr*z)

    # First derivatives
    qxu    = np.einsum('i,ij,j->', u, Q, x)
    rpe = r - qxu
    dQ_dlr  = rpe*z + dQ_dlr*(1 - lr*z)

    # Update
    AGENT.learning(x, u, r, x_, None)
    Q = Q + lr*rpe*z

# Check against autograd
np.linalg.norm(ag[0]-dlp_dlr)
np.linalg.norm(ag[1]-dlp_dB)
np.linalg.norm(ah-H)

np.linalg.norm(AGENT.hess_ - H)

np.linalg.norm(AGENT.actor.d_logprob['logits'] - dlp_dlogit)
np.linalg.norm(AGENT.d_logprob['inverse_softmax_temp'] - dlp_dB)
np.linalg.norm(AGENT.actor.d_logprob['action_values'] - dlp_dq)
np.linalg.norm(AGENT.actor.d_logprob['inverse_softmax_temp'] - np.dot(dlp_dlogit, dlogit_dB))
np.linalg.norm(AGENT.critic.dQ['learning_rate']-dQ_dlr)
np.linalg.norm(AGENT.critic.grad_Qx(x)-dq_dQ)
np.linalg.norm(AGENT.d_logprob['learning_rate']-dlp_dlr)
