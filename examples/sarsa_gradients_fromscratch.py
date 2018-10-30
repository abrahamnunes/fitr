import autograd.numpy as np
from autograd import jacobian, hessian
import fitr.utils as fu
import fitr.gradients as grad
import fitr.hessians as hess
from fitr.environments import DawTwoStep
from fitr.agents import SARSASoftmaxAgent

ntrials = 20
lr = 0.1; B = 2.; dc = 0.9; td=0.95
w = np.array([lr, B, dc, td])

task = DawTwoStep(rng=np.random.RandomState(532))
agent = SARSASoftmaxAgent(DawTwoStep(rng=np.random.RandomState(532)),
                          learning_rate=lr,
                          inverse_softmax_temp=B,
                          discount_factor=dc,
                          trace_decay=td,
                          rng=np.random.RandomState(236))
data = agent.generate_data(ntrials)

agent_inv = SARSASoftmaxAgent(DawTwoStep(rng=np.random.RandomState(532)),
                              learning_rate=lr,
                              inverse_softmax_temp=B,
                              discount_factor=dc,
                              trace_decay=td,
                              rng=np.random.RandomState(236))

X,U,R,X_,U_,DONE = data.unpack_tensor(task.nstates, task.nactions)

Q        = np.zeros((task.nactions, task.nstates))
DQ_lr    = np.zeros((task.nactions, task.nstates))
DQ_dc    = np.zeros((task.nactions, task.nstates))
DQ_td    = np.zeros((task.nactions, task.nstates))
D2Q_lr   = np.zeros((task.nactions, task.nstates))
D2Q_dc   = np.zeros((task.nactions, task.nstates))
D2Q_td   = np.zeros((task.nactions, task.nstates))
D2Q_lrdc = np.zeros((task.nactions, task.nstates))
D2Q_lrtd = np.zeros((task.nactions, task.nstates))
D2Q_dctd = np.zeros((task.nactions, task.nstates))

Drpe_Q  = np.zeros((task.nactions, task.nstates))
Drpe_lr = 0
Drpe_dc = 0
Drpe_td = 0

Dlp_lr = 0
Dlp_B  = 0
Dlp_dc = 0
Dlp_td = 0

D2lp_lr   = 0
D2lp_B    = 0
D2lp_dc   = 0
D2lp_td   = 0
D2lp_lrB  = 0
D2lp_lrdc = 0
D2lp_lrtd = 0
D2lp_Bdc  = 0
D2lp_Btd  = 0
D2lp_dctd = 0


L = 0
for t in range(R.size):
    x=X[0,t]; u=U[0,t]; r=R.flatten()[t]; x_=X_[0,t]; u_=U_[0,t]; done=DONE.flatten()[t]

    if done == 0: agent_inv.reset_trace()
    agent_inv.log_prob(x, u)
    agent_inv.learning(x, u, r, x_, u_)

    q = np.einsum('ij,j->i', Q, x)
    Dq_Q = x
    logits = B*q
    Dlogit_B = q
    Dlogit_q = B
    lp = logits - fu.logsumexp(logits)
    Dlp_logit = np.eye(logits.size) - np.tile(fu.softmax(logits).flatten(), [logits.size, 1])
    L += np.einsum('i,i->', u, lp)
    pu = fu.softmax(logits)
    du = u - pu
    dpu_dlogit = grad.softmax(logits)
    DQ_lr_state = np.dot(DQ_lr, x)
    DQ_dc_state = np.dot(DQ_dc, x)
    DQ_td_state = np.dot(DQ_td, x)
    dpu_lr = B*np.einsum('ij,j->i', dpu_dlogit, DQ_lr_state)
    dpu_B  = np.einsum('ij,j->i', dpu_dlogit, Dlogit_B)
    dpu_dc = B*np.einsum('ij,j->i', dpu_dlogit, DQ_dc_state)
    dpu_td = B*np.einsum('ij,j->i', dpu_dlogit, DQ_td_state)


    HB, Hq = hess.log_softmax(B, q)

    D2lp_lr   += B*np.dot(np.einsum('i,ij->j', du, D2Q_lr)  - np.einsum('i,ij->j', dpu_lr, DQ_lr), x)
    D2lp_dc   += B*np.dot(np.einsum('i,ij->j', du, D2Q_dc)  - np.einsum('i,ij->j', dpu_dc, DQ_dc), x)
    D2lp_td   += B*np.dot(np.einsum('i,ij->j', du, D2Q_td)  - np.einsum('i,ij->j', dpu_td, DQ_td), x)

    D2lp_lrdc += B*np.dot(np.einsum('i,ij->j', du, D2Q_lrdc) - np.einsum('i,ij->j', dpu_dc, DQ_lr), x)
    D2lp_lrtd += B*np.dot(np.einsum('i,ij->j', du, D2Q_lrtd) - np.einsum('i,ij->j', dpu_td, DQ_lr), x)
    D2lp_dctd += B*np.dot(np.einsum('i,ij->j', du, D2Q_dctd) - np.einsum('i,ij->j', dpu_td, DQ_dc), x)

    D2lp_B    += np.dot(u, HB)
    D2lp_lrB  += np.dot(du - B*dpu_B, DQ_lr_state)
    D2lp_Bdc  += np.dot(du - B*dpu_B, DQ_dc_state)
    D2lp_Btd  += np.dot(du - B*dpu_B, DQ_td_state)

    # Derivatives of log-probability with respect to learning rate
    Dlp_B += np.einsum('i,ij,j->', u, Dlp_logit, Dlogit_B)
    Dlp_q =  Dlp_logit*Dlogit_q

    #   First order
    Dq_lr = np.einsum('ij,j->i', DQ_lr, Dq_Q)
    Dlp_lr += np.einsum('i,ij,j->', u, Dlp_q, Dq_lr)
    # Derivatives of log-probability with respect to discount factor
    #   First order
    Dq_dc   = np.einsum('ij,j->i', DQ_dc, Dq_Q)
    Dlp_dc += np.einsum('i,ij,j->', u, Dlp_q, Dq_dc)

    # Derivatives of log-probability with respect to trace decay
    #   First order
    Dq_td   = np.einsum('ij,j->i',  DQ_td, Dq_Q)
    Dlp_td += np.einsum('i,ij,j->', u, Dlp_q, Dq_td)

    grad_ = np.array([Dlp_lr, Dlp_B, Dlp_dc, Dlp_td])
    hess_ = np.array([[D2lp_lr  , D2lp_lrB, D2lp_lrdc, D2lp_lrtd],
                      [D2lp_lrB ,   D2lp_B,  D2lp_Bdc,  D2lp_Btd],
                      [D2lp_lrdc, D2lp_Bdc, D2lp_dc  , D2lp_dctd],
                      [D2lp_lrtd, D2lp_Btd, D2lp_dctd,   D2lp_td]])

    #Reset trace
    if done == 0:
        D2z_dc   = np.zeros((task.nactions, task.nstates))
        D2z_td   = np.zeros((task.nactions, task.nstates))
        D2z_dctd = np.zeros((task.nactions, task.nstates))
        Dz_dc    = np.zeros((task.nactions, task.nstates))
        Dz_td    = np.zeros((task.nactions, task.nstates))
        z        = np.zeros((task.nactions, task.nstates))

    Drpe_Q     = dc*np.outer(u_, x_) - np.outer(u, x)

    # Compute derivatives
    D2z_dc = td*(2*Dz_dc + dc*D2z_dc)
    D2z_td = dc*(2*Dz_td + td*D2z_td)
    D2z_dctd = z + dc*Dz_dc + td*Dz_td + dc*td*D2z_dctd
    Dz_dc = td*(z + dc*Dz_dc)
    Dz_td = dc*(z + td*Dz_td)

    # Update trace
    z = np.outer(u, x) + dc*td*z

    # REWARD PREDICTION ERROR
    # Compute derivatives
    #   Second order

    D2rpe_lr   = np.sum(D2Q_lr*Drpe_Q)
    D2rpe_dc   = np.sum(D2Q_dc*Drpe_Q) + 2*np.einsum('i,ij,j->', u_, DQ_dc, x_)
    D2rpe_td   = np.sum(D2Q_td*Drpe_Q)
    D2rpe_lrdc = np.sum(D2Q_lrdc*Drpe_Q) + np.einsum('i,ij,j->', u_, DQ_lr, x_)
    D2rpe_lrtd = np.sum(D2Q_lrtd*Drpe_Q)
    D2rpe_dctd = np.sum(D2Q_dctd*Drpe_Q) + np.einsum('i,ij,j->', u_, DQ_td, x_)
    D2rpe_dcQ  = np.outer(u_, x_)

    #   First order
    Drpe_lr = np.sum(DQ_lr*Drpe_Q)
    Drpe_dc = np.sum(DQ_dc*Drpe_Q) + np.einsum('i,ij,j->', u_, Q, x_)
    Drpe_td = np.sum(DQ_td*Drpe_Q)

    # Compute RPE
    rpe = r + dc*np.einsum('i,ij,j->',u_,Q,x_) - np.einsum('i,ij,j->',u,Q,x)

    # Q PARAMETERS
    # Compute derivatives
    #   Second order
    D2Q_lr   += (2*Drpe_lr + lr*D2rpe_lr)*z
    D2Q_dc   += lr*(D2rpe_dc*z + Drpe_dc*(Dz_dc + D2z_dc))
    D2Q_td   += lr*(D2rpe_td*z + Drpe_td*(Dz_td + D2z_td))
    D2Q_lrdc += Drpe_dc*z + rpe*Dz_dc + lr*Drpe_lr*Dz_dc + lr*D2rpe_lrdc*z
    D2Q_lrtd += Drpe_td*z + rpe*Dz_td + lr*Drpe_lr*Dz_td + lr*D2rpe_lrtd*z
    D2Q_dctd += lr*(D2rpe_dctd*z + Drpe_dc*Dz_td + Drpe_td*Dz_dc + rpe*D2z_dctd)

    #   First order
    DQ_lr += (rpe + lr*Drpe_lr)*z
    DQ_dc += lr*(Drpe_dc*z + rpe*Dz_dc)
    DQ_td += lr*(Drpe_td*z + rpe*Dz_td)

    # Update value function
    Q += lr*rpe*z

# Functions to run through Autograd
def fQ(w):
    Q = np.zeros((task.nactions, task.nstates))
    L = 0
    for t in range(R.size):
        x=X[0,t]; u=U[0,t]; r=R.flatten()[t]; x_=X_[0,t]; u_=U_[0,t]; done=DONE.flatten()[t]
        logits = w[1]*np.einsum('ij,j->i', Q, x)
        lp = logits - fu.logsumexp(logits)
        L += np.einsum('i,i->', u, lp)
        #Reset trace
        if done == 0:
            z  = np.zeros((task.nactions, task.nstates))
        # Update trace
        z = np.outer(u, x) + w[2]*w[3]*z
        # Compute RPE
        rpe = r + w[2]*np.einsum('i,ij,j->', u_, Q, x_) - np.einsum('i,ij,j->', u, Q, x)
        # Update value function
        Q += w[0]*rpe*z
    return Q

def f(w):
    Q = np.zeros((task.nactions, task.nstates))
    L = 0
    for t in range(R.size):
        x=X[0,t]; u=U[0,t]; r=R.flatten()[t]; x_=X_[0,t]; u_=U_[0,t]; done=DONE.flatten()[t]
        logits = w[1]*np.einsum('ij,j->i', Q, x)
        lp = logits - fu.logsumexp(logits)
        L += np.einsum('i,i->', u, lp)
        #Reset trace
        if done == 0:
            z  = np.zeros((task.nactions, task.nstates))
        # Update trace
        z = np.outer(u, x) + w[2]*w[3]*z
        # Compute RPE
        rpe = r + w[2]*np.einsum('i,ij,j->', u_, Q, x_) - np.einsum('i,ij,j->', u, Q, x)
        # Update value function
        Q += w[0]*rpe*z
    return L

# Compare with autograd and the SARSASoftmax object.
print(' AUTOGRAD  \n\n ')
agH = hessian(f)(w)
print(agH)

print('\n\n FITR (RAW)  \n\n ')
print(hess_)

print('\n\n FITR (OBJECT) \n\n')
print(agent_inv.hess_)

agh = hessian(fQ)(w)
print(' AUTOGRAD  \n\n ')
print(' Learning rate \n')
print(agh[:,:,0,0])

print('\n\n Discount \n')
print(agh[:,:,2,2])

print('\n\n TD \n')
print(agh[:,:,3,3])

print('\n\n Discount/TD \n')
print(agh[:,:,2,3])
print(agh[:,:,3,2])

print('\n\n FITR \n\n')
print(' Learning rate \n')
print(D2Q_lr)
print('\n')
print(agent_inv.critic.hess_Q['learning_rate'])

print('\n\n Discount \n')
print(D2Q_dc)
print('\n')
print(agent_inv.critic.hess_Q['discount_factor'])

print('\n\n TD \n')
print(D2Q_td)
print('\n')
print(agent_inv.critic.hess_Q['trace_decay'])

print('\n\n Discount/TD ')
print(D2Q_dctd)
print('\n')
print(agent_inv.critic.hess_Q['discount_factor_trace_decay'])
