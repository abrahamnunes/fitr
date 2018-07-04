import numpy as np 
from scipy.special import logsumexp as scipy_logsumexp
from fitr.utils import logsumexp
from fitr.utils import relu
from fitr.utils import scale_data 
from fitr.utils import sigmoid 
from fitr.utils import softmax 
from fitr.utils import stable_exp

def test_logsumexp():
    x = np.arange(5)
    assert np.equal(logsumexp(x), scipy_logsumexp(x))

def test_relu():
    x  = np.linspace(-20, 20)
    y1 = relu(x)
    y2 = relu(x, a_max=10) 
    assert np.max(y1) == 20
    assert np.min(y1) == 0
    assert np.max(y2) == 10
    assert np.min(y2) == 0

def test_softmax():
    x = np.arange(10)
    p = softmax(x)
    assert p.sum() == 1
    assert np.logical_and(np.all(np.greater_equal(p, 0)), 
                          np.all(np.less_equal(p, 1)))
