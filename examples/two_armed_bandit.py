import numpy as np
import matplotlib.pyplot as plt
from fitr import generate_behavioural_data
from fitr.environments import TwoArmedBandit
from fitr.agents import RWSoftmaxAgent
from fitr.inference import mlepar
from fitr.utils import sigmoid
from fitr.utils import stable_exp
from fitr.criticism.plotting import actual_estimate
import fitr.gradients as grad

N = 50  # number of subjects
T = 200 # number of trials

# Generate synthetic data
data = generate_behavioural_data(TwoArmedBandit, RWSoftmaxAgent, N, T)

# Create log-likelihood function
def log_prob(w, D):
    lr  = sigmoid(w[0], a_min=-6, a_max=6)
    ist = stable_exp(w[1], a_min=-10, a_max=10)
    agent = RWSoftmaxAgent(TwoArmedBandit(), lr, ist)
    L = 0
    for t in range(D.shape[0]):
        x=D[t,:3]; u=D[t,3:5]; r=D[t,5]; x_=D[t,6:]
        agent.log_prob(x, u)
        agent.learning(x, u, r, x_, None)
    J = np.array([grad.sigmoid(w[0]), grad.exp(w[1])])
    return -agent.logprob_, -J*agent.grad_, 

# Fit model
res = mlepar(log_prob, data.tensor, nparams=2, maxstarts=5, jac=True)
X = res.transform_xmin([sigmoid, stable_exp])
idx = np.logical_and(np.logical_not(np.isnan(X[:,0])), np.less(X[:,1], 20))

# Criticism: Actual vs. Estimate Plots
lr_fig  = actual_estimate(data.params[idx,1], X[idx,0]); plt.show()
ist_fig = actual_estimate(data.params[idx,2], X[idx,1]); plt.show()
