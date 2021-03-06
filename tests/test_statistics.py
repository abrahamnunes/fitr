import numpy as np
import pandas as pd
import scipy.stats as ss
import fitr.utils as fu
from fitr.stats import distance
from fitr.stats import mean_ci
from fitr.stats import bic
from fitr.stats import lme
from fitr.stats import linear_regression
from fitr.stats import pearson_rho
from fitr.stats import spearman_rho
from fitr.stats.cluster import ami
from fitr.stats.cluster import completeness
from fitr.stats.cluster import homogeneity
from fitr.stats.cluster import silhouette
from fitr.stats.cluster import v_measure
from sklearn.datasets import make_blobs
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import silhouette_score

def test_ami():
    rng = np.random.RandomState(7254)
    y, _  = fu.make_onehot(rng.randint(5, size=100))
    y_, _ = fu.make_onehot(rng.randint(4, size=100))
    assert(np.equal(ami(y, y, inputs='onehot'), 1))

    skami = adjusted_mutual_info_score(np.argmax(y, 1), np.argmax(y_, 1))
    fami  = ami(y, y_, inputs='onehot')
    assert(np.linalg.norm(skami-fami) < 1e-8)

def test_silhouette():
    X, y = make_blobs(n_samples=300, centers=4, random_state=235)
    D = distance(X, X)
    fsil = silhouette(D, y)
    sksil = silhouette_score(D, y, metric='precomputed')
    assert(np.linalg.norm(fsil - sksil) < 1e-8)

def test_homogeneity_completeness_vmeasure():
    rng = np.random.RandomState(7254)
    y  = rng.randint(5, size=100)
    y_ = rng.randint(4, size=100)
    h = homogeneity(y, y_)
    c = completeness(y, y_)
    v = v_measure(y, y_)

    skh = homogeneity_score(y, y_)
    skc = completeness_score(y, y_)
    skv = v_measure_score(y, y_)

    assert(np.linalg.norm(h-skh) < 1e-8)
    assert(np.linalg.norm(c-skc) < 1e-8)
    assert(np.linalg.norm(v-skv) < 1e-8)

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

def test_linear_regression():
    X, y = make_regression(n_features=3, random_state=832)
    res = linear_regression(X, y)
    m = LinearRegression()
    m.fit(X, y)
    skcoef_ = np.hstack(([m.intercept_], m.coef_))
    assert(np.linalg.norm(skcoef_ - res.coef) < 1e-6)

    df = res.summary()
    assert(type(df)==pd.core.frame.DataFrame)

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
    rho, pval = spearman_rho(X, Y, 'pairwise')
    ssrho, sspval = ss.spearmanr(X, Y)
    assert(np.linalg.norm(ssrho[:3,3:] - rho) < 1e-6)
    assert(np.linalg.norm(sspval[:3,3:] - pval) < 1e-6)
