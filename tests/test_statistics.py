import numpy as np
import scipy.stats as ss
from fitr.stats import bic
from fitr.stats import lme 
from fitr.stats import pearson_rho
from fitr.stats import spearman_rho

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

