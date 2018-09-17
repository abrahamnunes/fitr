import numpy as np
from fitr import utils as fu
from fitr.environments import generate_behavioural_data
from fitr.environments import TwoArmedBandit
from fitr.environments import ReverseTwoStep
from fitr.environments import IGT
from fitr.agents import RWSoftmaxAgent
from fitr.criticism.plotting import actual_estimate
from scipy import optimize as op


task = TwoArmedBandit
data = generate_behavioural_data(task, RWSoftmaxAgent, 50, 200)
X1, U1, R, X2, U2, T = data.unpack_tensor(task().nstates, task().nactions)

def loglik(w, X, U, R, X_):
    w = fu.transform(w, [fu.sigmoid, np.exp]).flatten()
    q = RWSoftmaxAgent(task=task(),
                       learning_rate=w[0],
                       inverse_softmax_temp=w[1])
    ntrials = X.shape[0]
    for t in range(ntrials):
        q.log_prob(X[t], U[t])
        q.learning(X[t], U[t], R[t], X_[t], None)
    L = -q.logprob_
    return L, -q.grad_, -q.hess_

xhat = []
logprob = []
for i in range(data.nsubjects):
    f = lambda x: loglik(x, X1[i], U1[i], R[i], X2[i])[:-1]
    hess = lambda x: loglik(x, X1[i], U1[i], R[i], X2[i])[2]
    x0 = np.random.normal(0, 1, size=2)
    res = op.minimize(f, x0, jac=True, hess=hess, method='trust-exact')
    xhat.append(fu.transform(res.x, [fu.sigmoid, np.exp]).flatten())
    logprob.append(res.fun)
    print('FIT SUBJECT %s | LOGLIK %s' %(i, res.fun))

help(op.minimize)
xhat = np.array(xhat)
logprob = np.array(logprob)

f = actual_estimate(data.params[:,1], xhat[:,0])
f = actual_estimate(data.params[xhat[:,1] < 20 ,2], xhat[xhat[:,1] < 20 ,1])
