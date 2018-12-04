import autograd.numpy as np
from sklearn.metrics import log_loss as skl_logloss
from scipy.special import logsumexp as scipy_logsumexp
from fitr.utils import batch_softmax
from fitr.utils import batch_transform
from fitr.utils import bitflip
from fitr.utils import getquantile
from fitr.utils import I
from fitr.utils import log_loss
from fitr.utils import logsumexp
from fitr.utils import make_onehot
from fitr.utils import reduce_then_tile
from fitr.utils import relu
from fitr.utils import scale_data
from fitr.utils import sigmoid
from fitr.utils import sign
from fitr.utils import signinv
from fitr.utils import softmax
from fitr.utils import softmax_components
from fitr.utils import stable_exp
from fitr.utils import transform
from fitr.utils import tanh

def test_batch_softmax():
    X = np.random.randint(1, 10, size=(10, 5))
    p0 = np.stack(softmax(x_) for i, x_ in enumerate(X))
    p1 = batch_softmax(X, axis=1)
    p0_0 = np.stack(softmax(x_) for i, x_ in enumerate(X.T))
    p1_0 = batch_softmax(X, axis=0)
    assert np.all(np.equal(p0, p1))
    assert np.all(np.equal(p0_0.round(4), p1_0.round(4).T))

def test_batch_transform():
    X = np.random.normal(0, 5, size=(10, 2))
    Y = batch_transform(X, [sigmoid, stable_exp])
    assert(np.all(np.equal(X.shape, Y.shape)))
    assert(np.all(np.logical_and(np.greater_equal(Y[:,0], 0), np.less_equal(Y[:,0], 1))))
    assert(np.all(np.greater_equal(Y[:,0], 0)))

def test_bitflip():
    x = np.array([0, 1, 0, 1, 0])
    y = np.array([1, 0, 1, 0, 1])
    assert(np.all(np.equal(y, bitflip(x))))

def test_getquantile():
    x = np.linspace(0, 1, 100)
    y = getquantile(x, lower=0.25, upper=0.5)
    assert(np.equal(y.mean(), 0.25))

    y = getquantile(x, lower=0.25, upper=0.5, return_indices=True)
    assert(np.all(np.equal(y, np.arange(25, 50))))

def test_I():
    x = np.ones(5)
    assert np.all(np.equal(x, I(x)))

def test_logloss(): 
    rng = np.random.RandomState(457)
    y = np.random.binomial(1, p=0.6, size=20)
    yhat = np.random.uniform(0, 1, size=20)
    assert(np.linalg.norm(log_loss(y, yhat) - skl_logloss(y, yhat)) < 1e-10)

def test_logsumexp():
    x = np.arange(5)
    assert np.equal(logsumexp(x), scipy_logsumexp(x))

def test_make_onehot():
    rng = np.random.RandomState(23)
    X = rng.multinomial(1, pvals=[0.5, 0.5], size=20)
    y = np.argmax(X, axis=1)
    G,_ = make_onehot(y)
    assert(np.all(np.equal(X, G)))

def test_reduce_then_tile():
    X = np.random.randint(1, 10, size=(10, 5))
    p0 = np.stack(softmax(x_) for i, x_ in enumerate(X))
    max_x = reduce_then_tile(X, np.max, axis=1)
    exp_x = np.exp(X - max_x)
    sum_exp_x = reduce_then_tile(exp_x, np.sum, axis=1)
    p1 = exp_x/sum_exp_x
    assert np.all(np.equal(p0, p1))

def test_relu():
    x  = np.linspace(-20, 20)
    y1 = relu(x)
    y2 = relu(x, a_max=10)
    assert np.max(y1) == 20
    assert np.min(y1) == 0
    assert np.max(y2) == 10
    assert np.min(y2) == 0

def test_scale_data():
    x = np.arange(10).reshape(-1, 1)
    x = np.outer(x, np.ones(5))
    x = scale_data(x, with_mean=True, with_var=True)
    assert(np.all(np.equal(np.var(x, 0), np.ones(x.shape[1]))))

def test_sigmoid():
    x = np.linspace(-2, 2, 100)
    y = sigmoid(x)
    yidx = np.argsort(y)
    xidx = np.argsort(x)
    assert(np.all(np.logical_and(np.greater_equal(y, 0), np.less_equal(y, 1))))
    assert(np.all(np.equal(yidx, xidx)))

def test_sign():
    x = np.array([0, 1, 0, 1, 0])
    y = np.array([-1,1,-1, 1,-1])
    assert(np.all(np.equal(y, sign(x))))


def test_signinv():
    y = np.array([0, 1, 0, 1, 0])
    x = np.array([-1,1,-1, 1,-1])
    assert(np.all(np.equal(y, signinv(x))))

def test_softmax():
    x = np.arange(10)
    p = softmax(x)
    assert p.sum() == 1
    assert np.logical_and(np.all(np.greater_equal(p, 0)),
                          np.all(np.less_equal(p, 1)))

def test_softmax_partition_potential():
    x = np.arange(10).astype(np.float)
    B = 1.5
    potential, partition = softmax_components(B*x)
    component_sm = potential/partition
    original_sm = softmax(B*x)
    assert(np.all(component_sm == original_sm))


def test_transform():
    x  = np.array([0, 0, -10, 0, 55])
    x_ = transform(x, [sigmoid, tanh, relu, stable_exp, I])
    y  = np.array([0.5, 0, 0, 1, 55])
    assert np.all(np.equal(x_.flatten(), y))

def test_tanh():
    assert tanh(-1) < 0
    assert tanh(0) == 0
    assert tanh(1) > 0
    assert tanh(8, a_max=8) == tanh(10, a_max=8)
    assert tanh(-3, a_min=-3) == tanh(-7, a_min=-3)
