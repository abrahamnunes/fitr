# Simulating and Fitting a Two-Armed Bandit

``` python
import numpy as np
import matplotlib.pyplot as plt
from fitr import generate_behavioural_data
from fitr.environments import TwoArmedBandit
from fitr.agents import RWSoftmaxAgent
from fitr.inference import mlepar
from fitr.utils import sigmoid
from fitr.utils import relu
from fitr.criticism.plotting import actual_estimate

nsubjects = 50
ntrials = 200

# Generate synthetic data
data = generate_behavioural_data(Agent=RWSoftmaxAgent,
                                 environment=TwoArmedBandit(),
                                 nsubjects=nsubjects,
                                 ntrials=ntrials)

# Create log-likelihood function
def log_prob(w, D):
    learning_rate = sigmoid(w[0], a_min=-6, a_max=6)
    inverse_softmax_temp = relu(w[1], a_max=10)
    agent = RWSoftmaxAgent(TwoArmedBandit(),
                           learning_rate,
                           inverse_softmax_temp)
    L = 0
    for t in range(D.shape[0]):
        x  = D[t, :3]
        u  = D[t, 3:5]
        r  = D[t, 5]
        x_ = D[t, 6:]
        L += u@agent.log_prob(x)
        agent.learning(x, u, r, x_, None)
    return L

# Fit model
res = mlepar(f=log_prob,
             data=data.tensor,
             nparams=2,
             maxstarts=5)
X = res.transform_xmin([sigmoid, relu])

# Criticism
lr_fig  = actual_estimate(data.params[:,1], X[:,0]); plt.show()
ist_fig = actual_estimate(data.params[:,2], X[:,1]); plt.show()
```
