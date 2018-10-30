import numpy as np
from fitr import utils as fu
from fitr import gradients as grad
from fitr.environments import generate_behavioural_data
from fitr.environments import DawTwoStep
from fitr.agents import SARSASoftmaxAgent
from fitr.inference import mlepar
from fitr.criticism.plotting import actual_estimate

task = DawTwoStep
data = generate_behavioural_data(task, SARSASoftmaxAgent, 50, 200)
nx, nu = task().nstates, task().nactions


def reparam_jac(x):
    return np.diag(np.array([grad.sigmoid(x[0]),
                             grad.exp(x[1]),
                             grad.sigmoid(x[2])]))

def loglik(w, D):
    X1   = D[:,:nx]
    U1   = D[:,nx:nx+nu]
    R    = D[:,nx+nu].flatten()
    X2   = D[:,nx+nu+1:nx+nu+1+nx]
    U2   = D[:,nx+nu+1+nx:nx+nu+1+nx+nu]
    done = D[:,nx+nu+1+nx+nu:nx+nu+1+nx+nu+1].flatten()
    w = fu.transform(w, [fu.sigmoid, fu.stable_exp, fu.sigmoid]).flatten()
    J = reparam_jac(w)
    q = SARSASoftmaxAgent(task=task(),
                          learning_rate=w[0],
                          inverse_softmax_temp=w[1],
                          discount_factor=1.,
                          trace_decay=w[2])
    ntrials = X1.shape[0]
    for t in range(ntrials):
        if np.equal(done[t], 0):
            q.reset_trace()
            u2 = U2[t]
        else:
            u2 = np.zeros(nu)
        q.log_prob(X1[t], U1[t])
        q.learning(X1[t], U1[t], R[t], X2[t], u2)
    L = q.logprob_
    g = q.grad_[[0, 1, 3]]
    H = np.hstack((q.hess_[:,:2],q.hess_[:,-1].reshape(-1, 1)))
    H = np.vstack((H[:2,:], H[-1,:].reshape(1, -1)))
    return -L, -J@g, -J.T@H@J

res = mlepar(f=loglik,
             data=data.tensor,
             nparams=3,
             minstarts=2,
             maxstarts=10,
             maxstarts_without_improvement=2,
             init_sd=1,
             njobs=-1,
             jac=True,
             hess=True,
             method='trust-exact')


xhat = res.xmin[np.logical_not(np.any(np.isnan(res.xmin), axis=1)), :]
xhat = np.stack(fu.transform(xhat[i], [fu.sigmoid, fu.stable_exp, fu.sigmoid]).flatten() for i in range(xhat.shape[0]))
xtrue = data.params[np.logical_not(np.any(np.isnan(res.xmin), axis=1)), 1:]


f = actual_estimate(xtrue[:,0], xhat[:,0])
f = actual_estimate(xtrue[xhat[:,3]<20,3], xhat[xhat[:,3]<20,3])
f = actual_estimate(xtrue[xhat[:,3]<20,1], xhat[xhat[:,3]<20,1])
f = actual_estimate(xtrue[xhat[:,3]<20,2], xhat[xhat[:,3]<20,2])
