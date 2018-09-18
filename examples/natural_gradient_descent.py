import numpy as np
import fitr.utils as fu
from fitr import gradients as grad
from fitr.environments import TwoArmedBandit
from fitr.environments import generate_behavioural_data
from fitr.agents import RWStickySoftmaxAgent
from fitr.inference import mlepar
from fitr.criticism.plotting import actual_estimate

nsubjects = 100
ntrials = 500
task = TwoArmedBandit
data = generate_behavioural_data(task, RWStickySoftmaxAgent, nsubjects, ntrials)
nx, nu = task().nstates, task().nactions

clipped_exp = lambda x: fu.stable_exp(x, a_min=-5, a_max=5)

def reparam_jac(x):
    return np.diag(np.array([grad.sigmoid(x[0]), grad.exp(x[1]), x[2]]))

def loglik(w, D):
    X1, U1, R, X2  = D[:,:nx], D[:,nx:nx+nu], D[:,nx+nu], D[:,nx+nu+1:nx+nu+1+nx]
    w = fu.transform(w, [fu.sigmoid, clipped_exp, fu.I]).flatten()
    J = reparam_jac(w)
    q = RWStickySoftmaxAgent(task=task(),
                       learning_rate=w[0],
                       inverse_softmax_temp=w[1],
                       perseveration=w[2])
    ntrials = X1.shape[0]
    for t in range(ntrials):
        q.log_prob(X1[t], U1[t])
        q.learning(X1[t], U1[t], R[t], X2[t], None)
    L = q.logprob_
    return -L, -J@q.grad_, -J.T@q.hess_@J

def natural_gradient_descent(f,
                             i,
                             data,
                             nparams,
                             jac,
                             hess,
                             minstarts=2,
                             maxstarts=10,
                             maxstarts_without_improvement=3,
                             init_sd=2):

i = 4

ftol = 1e-4
xtol = 1e-4
maxiter = 25
ng_xhat = []
step_size_decay = 0.95
momentum = 0.1
for i in range(nsubjects):
    f = lambda x: loglik(x, data.tensor[i])
    niter = 0
    nprintiter = 5
    L0 = np.inf
    done = False
    x = None
    Linit = np.inf
    for j in range(10):
        xinit = np.random.normal(0, 2, size=2)
        L, _, _ = f(xinit)
        if L < Linit:
            x = xinit
    dx = np.inf
    step_size = 1.
    natural_gradient0 = np.zeros(x.size)
    while not done:
        niter += 1
        L, g, G = f(x)
        #U,S,V = np.linalg.svd(G)
        #G = U@(np.diag(np.abs(S))@V.T)
        D,B = np.linalg.eig(G)
        G = np.einsum('ij,jk,kl->il', B, np.diag(np.abs(D)), B)
        Ginv = np.linalg.inv(G)
        natural_gradient = np.einsum('ij,j->i', Ginv, g)
        x_ = x - step_size*natural_gradient - momentum*natural_gradient0

        dL = np.linalg.norm(L-L0); L0 = L
        dx = np.einsum('ij,i,j->', G, x-x_, x-x_); x = x_
        gnorm = np.linalg.norm(natural_gradient)
        x = x_
        step_size = step_size_decay*step_size
        natural_gradient0 = natural_gradient

        if niter % nprintiter == 0:
            print('Subject %s | Iteration %s | -lp_ = %s | dL = %s | dx = %s | lr = %s' %(i, niter, np.round(L, 3), np.round(dL, 3), np.round(dx, 3), np.round(step_size, 3)))
        if np.less(dL, ftol):
            done = True
            ng_xhat.append(x)
            print('Subject %s Fit Successfully after %s Iterations' %(i, niter))

        if np.less(step_size, 1e-3):
            done = True
            ng_xhat.append(x)
            print('Subject %s Fit Successfully after %s Iterations' %(i, niter))

        if niter >= maxiter:
            done = True
            ng_xhat.append(x)
            print('Subject %s Not fit successfully after %s iterations. Maximum number of iterations exceeded.' %(i, niter))

ng_xhat = np.array(ng_xhat)
ng_xhat_trans = fu.batch_transform(ng_xhat, [fu.sigmoid, fu.stable_exp])

nlog_prob = lambda x: loglik(x, data.tensor[i])[:-1]

res = mlepar(f=loglik,
             data=data.tensor,
             nparams=3,
             minstarts=2,
             maxstarts=6,
             maxstarts_without_improvement=2,
             init_sd=2,
             njobs=-1,
             jac=True,
             hess=True,
             method='trust-exact')




xhat = res.transform_xmin([fu.sigmoid, clipped_exp])
idx = np.logical_and(np.logical_not(np.isnan(xhat[:, 0])), np.less(xhat[:,1], 20))

f = actual_estimate(data.params[:,1], ng_xhat_trans[:,0])
f = actual_estimate(data.params[ng_xhat_trans[:,1] <20,2], ng_xhat_trans[ng_xhat_trans[:,1] <20,1])


f = actual_estimate(data.params[idx,1], xhat[idx,0])
f = actual_estimate(data.params[idx,2], xhat[idx,1])
