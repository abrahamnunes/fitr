import numpy as np 
from fitr import generate_behavioural_data
from fitr.data import merge_behavioural_data
from fitr.data import fast_unpack
from fitr.environments import TwoArmedBandit
from fitr.agents import RWSoftmaxAgent

def test_bdf():
    data1 = generate_behavioural_data(TwoArmedBandit, RWSoftmaxAgent, 10, 20)
    data2 = generate_behavioural_data(TwoArmedBandit, RWSoftmaxAgent, 10, 20)
    data = merge_behavioural_data([data1, data2])


def test_cooccurrence_matrix():
    data1 = generate_behavioural_data(TwoArmedBandit, RWSoftmaxAgent, 10, 20)
    data1.make_behavioural_ngrams(2)
    data1.make_cooccurrence_matrix(2)

def test_fast_unpack():
    ranges = [np.arange(2), np.arange(2)+2, 4, 5+np.arange(2)]
    D = np.outer(np.ones(5), np.arange(7))
    x, u, r, x_ = fast_unpack(D[0], ranges)
    assert(np.all(np.equal(x, D[0, 0:2])))
    assert(np.all(np.equal(u, D[0,2:4])))
    assert(np.all(np.equal(r, D[0,4])))
    assert(np.all(np.equal(x_, D[0,5:])))
