import numpy as np
from fitr.inference import mlepar
import pytest

def test_bms_verbose():
    X = np.random.uniform(0, 200, size=(100, 10))
    pxp, xp, bor, q_m, alpha, f0, f1, niter = bms(X, verbose=True)
    assert np.equal(pxp.sum(), 1)
    assert np.equal(xp.sum(), 1)
    assert (0 <= bor <= 1)
    assert np.equal(q_m.sum(1), 1)
    assert np.all(np.greater(alpha, 0))
    assert np.greater_equal(f0, 0)
    assert np.greater_equal(f1, 0)
    assert np.greater(niter, 0)

def test_bms_nptverbose():
    X = np.random.uniform(0, 200, size=(100, 10))
    pxp, xp, bor, q_m, alpha, f0, f1, niter = bms(X, verbose=False)
    assert np.equal(pxp.sum(), 1)
    assert np.equal(xp.sum(), 1)
    assert (0 <= bor <= 1)
    assert np.equal(q_m.sum(1), 1)
    assert np.all(np.greater(alpha, 0))
    assert np.greater_equal(f0, 0)
    assert np.greater_equal(f1, 0)
    assert np.greater(niter, 0)

def test_failure():
    """
    Test that should have at least some failures to do loglik function returning random values
    """
    def bad_loglik(x, D):
        i = x[0]
        j = x[1]
        return 10*i*np.random.random()-0.5 + 10*j*np.random.random()-0.5

    dummy_subject_data = np.random.random((50,5,2))

    with pytest.raises(ValueError):
        return mlepar(bad_loglik, dummy_subject_data, nparams=2,maxstarts=2)
