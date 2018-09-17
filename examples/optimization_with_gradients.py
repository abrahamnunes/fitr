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


task = TwoArmedBandit
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
             init_sd=2,
             njobs=-1,
             jac=True,
             hess=True,
             method='trust-ncg')
xhat = fu.transform(res.xmin[np.logical_not(np.any(np.isnan(res.xmin), axis=1)), :], [fu.sigmoid, np.exp])
xtrue = data.params[np.logical_not(np.any(np.isnan(res.xmin), axis=1)), 1:]

xhat = []
logprob = []
for i in range(data.nsubjects):
    f = lambda x: loglik(x, X1[i], U1[i], R[i], X2[i])[:-1]
    hess = lambda x: loglik(x, X1[i], U1[i], R[i], X2[i])[2]
    x0 = np.random.normal(0, 1, size=2)
    res = op.minimize(f, x0, jac=True, hess=hess, method='trust-krylov')
    xhat.append(fu.transform(res.x, [fu.sigmoid, np.exp]).flatten())
    logprob.append(res.fun)
    print('FIT SUBJECT %s | LOGLIK %s' %(i, res.fun))

help(op.minimize)
xhat = np.array(xhat)
logprob = np.array(logprob)

f = actual_estimate(xtrue[:,0], xhat[:,0])
f = actual_estimate(data.params[xhat[:,1] < 20 ,2], xhat[xhat[:,1] < 20 ,1])
