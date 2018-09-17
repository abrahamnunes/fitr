import cProfile
import numpy as np
from fitr import utils as fu
from fitr.environments import TwoArmedBandit
from fitr.agents import RWSoftmaxAgent

task = TwoArmedBandit(rng=np.random.RandomState(743))
agent = RWSoftmaxAgent(task, learning_rate=0.4, inverse_softmax_temp=2.6)

# Generate data
data = agent.generate_data(ntrials=1000)
unpacked_data = data.unpack_tensor(task.nstates, task.nactions)
X, U, R, X_, _, _ = [np.squeeze(di) for di in unpacked_data]

def f():
    agent = RWSoftmaxAgent(task, learning_rate=0.4, inverse_softmax_temp=2.6)
    for t in range(X.shape[0]):
        agent.log_prob(X[t], U[t])
        agent.learning(X[t], U[t], R[t], X_[t], None)
    return agent.logprob_


cProfile.run('agent.log_prob(X[5], U[5])', sort='time')
cProfile.run('f()', sort='time')
