import autograd.numpy as np
from autograd import jacobian, hessian, elementwise_grad
import fitr.utils as fu
import fitr.gradients as grad
from fitr.environments import DawTwoStep
from fitr.data import BehaviouralData
from fitr.agents.agents import TwoStepStickySoftmaxSARSABellmanMaxAgent

ntrials = 201
env = DawTwoStep(rng = np.random.RandomState(436))

data = BehaviouralData(1)

lr1 = 0.1; lr2 = 0.1; B1 = 2.; persev=0.0; B2=2; td=0.95; w = 0.1
par = np.array([lr1, lr2, B1, B2, td, w])
rng = np.random.RandomState(32)
T   = env.T
Qmf = np.zeros((env.nactions, env.nstates))
Q   = np.zeros((env.nactions, env.nstates))

agent = TwoStepStickySoftmaxSARSABellmanMaxAgent(env, lr1, lr2, B1, B2, td, w, persev)

data.add_subject(subject_index=0, parameters=par, subject_meta=[])
for t in range(ntrials):
    x  = env.observation()
    u = agent.action_step1(x)
    x_, r, _ = env.step(u)
    u_ = agent.action_step2(x_)
    xf, r_, _ = env.step(u_)
    agent.learning(x, u, x_, u_, r_)
    data.update(0, np.hstack([x, u, x_, u_, r_]))
data.make_tensor_representations()


X, U, X_, U_, R = data.unpack_tensor(env.nstates, env.nactions, get='sasar')
X = np.squeeze(X)
U = np.squeeze(U)
X_ = np.squeeze(X_)
U_ = np.squeeze(U_)
R = R.flatten()
agent_lp = TwoStepStickySoftmaxSARSABellmanMaxAgent(env, lr1, lr2, B1, B2, td, w, persev)
for t in range(R.size):
    x = X[t]; u=U[t]; r=R[t]; x_=X_[t]; u_=U_[t];
    agent_lp.log_prob(x, u, x_, u_)
    agent_lp.learning(x, u, x_, u_, r)



# Parameter inference with hand-computed gradients/hessian
L          = 0
Dlp_lr1    = 0
Dlp_lr2    = 0
Dlp_td     = 0
Dlp_B1     = 0
Dlp_B2     = 0
Dlp_w      = 0
Dlp_logit1 = np.zeros(env.nactions)
Dlp_logit2 = np.zeros(env.nactions)
Dlogit1_B1 = np.zeros(env.nactions)
Dlogit2_B2 = np.zeros(env.nactions)
Dlogit1_q1 = B1
Dlogit2_q2 = B2
Dq1_Qmb    = np.zeros((env.nactions, env.nstates))
Dq2_Qmf    = np.zeros((env.nactions, env.nstates))
DQ_Qmb     = w
DQ_Qmf     = 1-w
DQmb_Qmf   = np.zeros((env.nactions, env.nstates))

DQmf_lr1 = np.zeros((env.nactions, env.nstates))
DQmf_lr2 = np.zeros((env.nactions, env.nstates))
DQmf_td  = np.zeros((env.nactions, env.nstates))
DQmb_lr1 = np.zeros((env.nactions, env.nstates))
DQmb_lr2 = np.zeros((env.nactions, env.nstates))
DQ_lr1   = np.zeros((env.nactions, env.nstates))
DQ_lr2   = np.zeros((env.nactions, env.nstates))
DQ_w     = np.zeros((env.nactions, env.nstates))

X, U, X_, U_, R = data.unpack_tensor(env.nstates, env.nactions, get='sasar')
X = np.squeeze(X)
U = np.squeeze(U)
X_ = np.squeeze(X_)
U_ = np.squeeze(U_)
R = R.flatten()

T   = env.T; T[:,1:,1:] = 0
Qmf = np.zeros((env.nactions, env.nstates))
Qmb = np.zeros((env.nactions, env.nstates))
Q   = np.zeros((env.nactions, env.nstates))
L = 0
ulast = np.zeros(env.nactions)
for t in range(R.size):
    x = X[t]; u=U[t]; r=R[t]; x_=X_[t]; u_=U_[t];
    q          = np.einsum('ij,j->i', Q, x)
    q_         = np.einsum('ij,j->i', Qmf, x_)
    logits1    = B1*q
    logits2    = B2*q_
    pu1        = fu.softmax(logits1)
    pu2        = fu.softmax(logits2)
    Dlp_logit1 = np.eye(q.size)  - np.tile(grad.logsumexp(B1*q), [q.size, 1])
    Dlp_logit2 = np.eye(q_.size) - np.tile(grad.logsumexp(B2*q_), [q_.size, 1])
    Dlogit1_q1 = B1
    Dlogit2_q2 = B2
    Dlogit1_B1 = q
    Dlogit2_B2 = q_
    Dlp_q1     = B1*np.eye(q.size)  - np.tile(B1*grad.logsumexp(B1*q), [q.size, 1])
    Dlp_q2     = B2*np.eye(q_.size)  - np.tile(B2*grad.logsumexp(B2*q_), [q_.size, 1])
    Dq1_Q      = x
    Dq2_Q      = x_
    Dq2_Qmf    = x_
    DQ_w       = Qmb - Qmf

    L += np.dot(u, logits1 - fu.logsumexp(logits1))
    L += np.dot(u_, logits2 - fu.logsumexp(logits2))

    Dlp_lr1 += np.dot(u,  np.einsum('ij,jk,k->i', Dlp_q1, DQ_lr1, Dq1_Q))
    Dlp_lr1 += np.dot(u_, np.einsum('ij,jk,k->i', Dlp_q2, DQmf_lr1, Dq2_Qmf))
    Dlp_lr2 += np.dot(u,  np.einsum('ij,jk,k->i', Dlp_q1, DQ_lr2, Dq1_Q))
    Dlp_lr2 += np.dot(u_, np.einsum('ij,jk,k->i', Dlp_q2, DQmf_lr2, Dq2_Qmf))
    Dlp_w   += np.dot(u,  np.dot(Dlp_q1, np.dot(DQ_w, Dq1_Q)))
    Dlp_td  += np.dot(u,  np.einsum('ij,jk,k->i', Dlp_q1, (DQ_Qmb*DQmb_Qmf + DQ_Qmf)*DQmf_td, Dq1_Q))
    Dlp_td  += np.dot(u_, np.einsum('ij,jk,k->i', Dlp_q2, DQmf_td, Dq2_Qmf))

    Dlp_B1  += u@Dlp_logit1@Dlogit1_B1
    Dlp_B2  += u_@Dlp_logit2@Dlogit2_B2

    # UPDATE VALUE FUNCTION
    z1   = np.outer(u, x)
    z2   = np.outer(u_, x_) + td*z1

    DQmf_lr1 += ((u_.T@Qmf@x_) - (u.T@Qmf@x))*z1 + lr1*(((u_.T@DQmf_lr1@x_) - (u.T@DQmf_lr1@x))*z1 - (u_.T@DQmf_lr1@x_)*z2)
    DQmf_lr2 += lr1*((u_.T@DQmf_lr2@x_) - (u.T@DQmf_lr2@x))*z1 + (r - (u_.T@Qmf@x_) - lr2*(u_.T@DQmf_lr2@x_))*z2
    DQmf_td  += lr1*((u_.T@DQmf_td@x_) - (u.T@DQmf_td@x))*z1 + lr2*((r - (u_.T@Qmf@x_))*z1 - (u_.T@DQmf_td@x_)*z2)
    Qmf      += lr1*(u_.T@Qmf@x_)*z1 - lr1*(u.T@Qmf@x)*z1 + lr2*r*z2 - lr2*(u_.T@Qmf@x_)*z2

    DmaxQmf_Qmf = grad.matrix_max(Qmf, axis=0)
    DQmb_maxQmf = np.tile(np.sum(np.sum(T, axis=0), axis=1), [q.size, 1])
    DQmb_Qmf    = DQmb_maxQmf*DmaxQmf_Qmf
    DQ_Qmf      = 1 - w + w*DQmb_Qmf

    DQmb_lr1 = np.einsum('ijk,j->ik', T, np.einsum('ij,ij->j', DmaxQmf_Qmf, DQmf_lr1))
    DQ_lr1 = w*DQmb_lr1 + (1-w)*DQmf_lr1 #DQ_Qmf*DQmf_lr1
    DQmb_lr2 = np.einsum('ijk,j->ik', T, np.einsum('ij,ij->j', DmaxQmf_Qmf, DQmf_lr2))
    DQ_lr2 = w*DQmb_lr2 + (1-w)*DQmf_lr2
    DQ_td  = DQ_Qmf*DQmf_td

    maxQmf = np.max(Qmf, axis=0)
    Qmb    = np.einsum('ijk,j->ik', T, maxQmf)
    Q      = w*Qmb + (1-w)*Qmf

    ulast=U[t]

L


agent_lp.logprob_


Dlp_lr1
Dlp_lr2
Dlp_B1
Dlp_B2
Dlp_td
Dlp_w

agent_lp.d_logprob['learning_rate_1']
agent_lp.d_logprob['learning_rate_2']
agent_lp.d_logprob['inverse_softmax_temp_1']
agent_lp.d_logprob['inverse_softmax_temp_2']
agent_lp.d_logprob['trace_decay']
agent_lp.d_logprob['mb_weight']
