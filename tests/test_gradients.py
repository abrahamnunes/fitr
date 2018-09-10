import numpy as np
from fitr import utils
from fitr import gradients as grad
from scipy.optimize import approx_fprime

def test_logsumexp():
    x = np.array([1, 0, 0])
    eps = np.ones(x.size)*1e-8
    f = lambda x: utils.logsumexp(x)
    g = lambda x: grad.logsumexp(x)
    grad_exact = g(x)
    grad_approx= approx_fprime(x, f, eps)
    err = np.linalg.norm(grad_exact-grad_approx)
    assert(err < 1e-5)
