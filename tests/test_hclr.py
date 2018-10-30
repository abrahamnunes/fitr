import numpy as np
from fitr.hclr import HCLR

def test_initialization():
    nsubjects = 100
    ntrials = 200
    nfeatures = 4
    ncovariates=5
    filter_size=10
    add_intercept=True
    loading_matrix_scale = 5.

    rng = np.random.RandomState(9723)
    X = []
    y = rng.binomial(1, p=0.5, size=(nsubjects, ntrials))
    Z = rng.normal(size=(nsubjects, ncovariates))
    for i in range(nsubjects):
        Xi = rng.multinomial(1, pvals=np.ones(nfeatures)/nfeatures, size=ntrials)
        y_ = 2*y[i] - 1
        y_ = np.tile(y_.reshape(-1, 1), [1, Xi.shape[1]])
        Xi = Xi*y_
        X.append(Xi)

    V = np.array([[0., 1., -1., -1., 1.], [0., 1., 1., -1., -1.]])


    m = HCLR(X, y, Z, V, filter_size, loading_matrix_scale, add_intercept)
    assert(m.nsubjects == nsubjects)
    assert(m.loading_matrix_scale == loading_matrix_scale)
    assert(m.ntrials == ntrials-filter_size-1)
    assert(m.naxes == V.shape[0])
    assert(m.nfeatures == (int(nfeatures*filter_size) + int(add_intercept)))
    assert(np.all(m.v_bias.flatten() == V[:,0]))

    assert(m.data['n_s'] == nsubjects)
    assert(m.data['n_t'] == ntrials-filter_size-1)
    assert(m.data['n_f'] == (int(nfeatures*filter_size) + int(add_intercept)))
    assert(m.data['n_c'] == ncovariates)
    assert(m.data['K_scale'] == loading_matrix_scale)
    assert(m.data['n_v'] == V.shape[0])
    assert(np.all(m.data['X'] == m.X))
    assert(np.all(m.data['y'] == m.y))
    assert(np.all(m.data['Z'] == m.Z))

    # Check that the data flattening is working correctly
    X = np.array(X)
    x1flat = X[0, :filter_size,:].flatten()
    x1flat = np.hstack(([1], x1flat))
    assert(np.all(m.X[0,0,:] == x1flat))

    # Check another value just to be sure
    x20flat = X[0, 20:filter_size+20,:].flatten()
    x20flat = np.hstack(([1], x20flat))
    assert(np.all(m.X[0,20,:] == x20flat))

def test_stanfit():
    nsubjects = 100
    ntrials = 200
    nfeatures = 4
    ncovariates=5
    filter_size=10
    add_intercept=True
    loading_matrix_scale = 5.
    nchains = 2
    niter = 50

    rng = np.random.RandomState(9723)
    X = []
    y = rng.binomial(1, p=0.5, size=(nsubjects, ntrials))
    Z = rng.normal(size=(nsubjects, ncovariates))
    for i in range(nsubjects):
        Xi = rng.multinomial(1, pvals=np.ones(nfeatures)/nfeatures, size=ntrials)
        y_ = 2*y[i] - 1
        y_ = np.tile(y_.reshape(-1, 1), [1, Xi.shape[1]])
        Xi = Xi*y_
        X.append(Xi)

    V = np.array([[0., 1., -1., -1., 1.], [0., 1., 1., -1., -1.]])


    m = HCLR(X, y, Z, V, filter_size, loading_matrix_scale, add_intercept)
    m.fit(nchains=nchains, niter=niter, warmup=0, seed=234)
