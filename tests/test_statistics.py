import numpy as np
import scipy.stats as ss
import fitr.utils as fu
from fitr.stats import distance
from fitr.stats import mean_ci
from fitr.stats import bic
from fitr.stats import lme 
from fitr.stats import pearson_rho
from fitr.stats import spearman_rho
from sklearn.metrics import pairwise_distances

def test_distance():
    rng = np.random.RandomState(235)
    X = rng.normal(size=(100, 200))
    Y = rng.normal(size=(100, 200))

    # Euclidean
    skd = pairwise_distances(X, Y)
    fud = distance(X, Y)
    assert(np.linalg.norm(skd-fud) < 1e-8)

    # Manhattan
    skd = pairwise_distances(X, Y, metric='manhattan')
    fud = distance(X, Y, metric='manhattan')
    assert(np.linalg.norm(skd-fud) < 1e-8)

    # Chebyshev
    skd = pairwise_distances(X, Y, metric='chebyshev')
    fud = distance(X, Y, metric='chebyshev')
    assert(np.linalg.norm(skd-fud) < 1e-8)

    # Canberra 
    skd = pairwise_distances(X, Y, metric='canberra')
    fud = distance(X, Y, metric='canberra')
    assert(np.linalg.norm(skd-fud) < 1e-8)

def test_mean_ci():
    rng = np.random.RandomState(235)
    n = 100
    m = 200
    alpha = 0.05
    e_ci = np.abs(ss.norm.ppf(alpha/2))
    X = rng.normal(0, 1, size=(n, m))
     
    # Test along first axis
    X0 = fu.scale_data(X, axis=0)
    mu, l, u = mean_ci(X0, axis=0, alpha=alpha)
    assert(np.linalg.norm(mu) < 1e-8)
    assert(np.linalg.norm(np.abs(l)-(e_ci/np.sqrt(n))) < 1e-8)
    

    # Test along second axis
    X1 = fu.scale_data(X, axis=1)
    mu, l, u = mean_ci(X1, axis=1, alpha=alpha)
    assert(np.linalg.norm(mu) < 1e-8)
    assert(np.linalg.norm(np.abs(l)-(e_ci/np.sqrt(m))) < 1e-8)
    

def test_pearson_rho():
    rng = np.random.RandomState(523)
    X = rng.normal(size=(20, 3))
    Y = rng.normal(size=(20, 3))
    rho, pval = pearson_rho(X, Y)

    # Test column-wise
    for j in range(X.shape[1]):
        ssrho, sspval = ss.pearsonr(X[:,j], Y[:,j])
        assert(np.linalg.norm(ssrho - rho[j]) < 1e-6)
        assert(np.linalg.norm(sspval - pval[j]) < 1e-6)

    # Test pairwise 
    rho, pval = pearson_rho(X, Y, 'pairwise')
    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            ssrho, sspval = ss.pearsonr(X[:,i], Y[:,j])
            assert(np.linalg.norm(ssrho - rho[i,j]) < 1e-6)
            assert(np.linalg.norm(sspval - pval[i,j]) < 1e-6)

def test_spearman_rho():
    rng = np.random.RandomState(523)
    X = rng.normal(size=(20, 3))
    Y = rng.normal(size=(20, 3))
    rho, pval = spearman_rho(X, Y)

    # Test column-wise
    for j in range(X.shape[1]):
        ssrho, sspval = ss.spearmanr(X[:,j], Y[:,j])
        assert(np.linalg.norm(ssrho - rho[j]) < 1e-6)
        assert(np.linalg.norm(sspval - pval[j]) < 1e-6)

    # Test pairwise 
    rho, pval = pearson_rho(X, Y, 'pairwise')
    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            ssrho, sspval = ss.spearmanr(X[:,i], Y[:,j])
            assert(np.linalg.norm(ssrho - rho[i,j]) < 1e-6)
            assert(np.linalg.norm(sspval - pval[i,j]) < 1e-6)

