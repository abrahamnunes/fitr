import numpy as np
from fitr import utils as fu
from fitr import gradients as grad
from fitr.environments import generate_behavioural_data
from fitr.environments import TwoArmedBandit
from fitr.environments import ReverseTwoStep
from fitr.environments import IGT
from fitr.agents import RWSoftmaxAgent
from fitr.inference import mlepar
from fitr.criticism.plotting import actual_estimate
from scipy import optimize as op


task = IGT
data = generate_behavioural_data(task, RWSoftmaxAgent, 50, 200)
nx, nu = task().nstates, task().nactions

def reparam_jac(x):
    return np.diag(np.array([grad.sigmoid(x[0]), x[1]]))

def loglik(w, D):
    X1, U1, R, X2  = D[:,:nx], D[:,nx:nx+nu], D[:,nx+nu], D[:,nx+nu+1:nx+nu+1+nx]
    w = fu.transform(w, [fu.sigmoid, np.exp]).flatten()
    J = reparam_jac(w)
    q = RWSoftmaxAgent(task=task(),
                       learning_rate=w[0],
                       inverse_softmax_temp=w[1])
    ntrials = X1.shape[0]
    for t in range(ntrials):
        q.log_prob(X1[t], U1[t])
        q.learning(X1[t], U1[t], R[t], X2[t], None)
    L = q.logprob_
    return -L, -J@q.grad_, -J.T@q.hess_@J

res = mlepar(f=loglik,
             data=data.tensor,
             nparams=2,
             minstarts=2,
             maxstarts=15,
             maxstarts_without_improvement=5,
             init_sd=2,
             njobs=-1,
             jac=True,
             hess=True,
             method='trust-krylov')

res.xmin

xhat = res.xmin[np.logical_not(np.any(np.isnan(res.xmin), axis=1)), :]
xhat = np.stack(fu.transform(xhat[i], [fu.sigmoid, fu.stable_exp]).flatten() for i in range(xhat.shape[0]))
xtrue = data.params[np.logical_not(np.any(np.isnan(res.xmin), axis=1)), 1:]

f = actual_estimate(xtrue[:,0], xhat[:,0])
f = actual_estimate(xtrue[xhat[:,1]<20,1], xhat[xhat[:,1]<20,1])
