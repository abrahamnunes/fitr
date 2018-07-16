import numpy as np
from scipy.stats import dirichlet
from scipy.special import digamma
from scipy.special import gammaln

from fitr.utils import batch_softmax
from fitr.utils import reduce_then_tile

def bms(L, ftol=1e-12, nsamples=1000000, rng=np.random.RandomState(), verbose=True):
    """ Implements variational Bayesian Model Selection as per Rigoux et al. (2014).

    Arguments:

        L: `ndarray((nsubjects, nmodels))`. Log model evidence
        ftol: `float`. Threshold for convergence of prediction error
        nsamples: `int>0`. Number of samples to draw from Dirichlet distribution for computation of exceedence probabilities
        rng: `np.random.RandomState`
        verbose: `bool (default=True)`. If `False`, no output provided.

    Returns:
        pxp: `ndarray(nmodels)`. Protected exceedance probabilities
        xp: `ndarray(nmodels)`. Exceedance probabilities
        bor: `ndarray(nmodels)`. Bayesian Omnibus Risk
        pe: `ndarray(niter)`. Prediction error time series throughout optimization
        q_m: `ndarray((nsubjects, nmodels))`. Posterior distribution over models for each subject
        alpha: `ndarray(nmodels)`. Posterior estimates of Dirichlet parameters
        f0: `float`. Free energy of null model
        f1: `float`. Free energy of alternative model
        niter: `int`. Number of iterations of posterior optimization

    Examples:

    Assuming one is given a matrix of (log-) model evidence values `L` of type `ndarray((nsubjects, nmodels))`,

    ```
    from fitr.inference import spm_bms

    pxp, xp, bor, q_m, alpha, f0, f1, niter = bms(L)
    ```

    Todos:

    - [ ] Add notes on derivation

    """
    nsubjects, nmodels = L.shape

    if verbose:
        print('========= BAYESIAN MODEL SELECTION =========\n' +
              'Number of models: ' + str(nmodels) + '\n' +
              'Subjects: ' + str(nsubjects) + '\n' +
              'Convergence limit: ' + str(ftol) + '\n' +
              '=============================================\n')

    alpha0 = np.ones(nmodels)
    q_m, alpha, niter = infer_dirichlet_multinomial_posterior(L, alpha0, ftol, verbose)
    r   = dirichlet_mean(alpha)
    xp  = dirichlet_exceedance_probability(alpha, nsamples, rng)
    f0  = free_energy_null(L)
    f1  = free_energy_alternative(L, q_m, alpha, alpha0)
    bor = bayesian_omnibus_risk(f0, f1)
    pxp = protected_exceedance_probability(xp, bor)
    return pxp, xp, bor, q_m, alpha, f0, f1, niter

def infer_dirichlet_multinomial_posterior(L, a, ftol=1e-12, verbose=True):
    """ Computes the poseterior distribution of Dirichlet-Multinomial according to the variational Bayesian procedure specified by Rigoux et al. (2014).

    Arguments:

        L: `ndarray((nsubjects, nmodels))`. Log-model evidence
        a: `ndarray(nmodels)`. Prior estimates of Dirichlet model parameters
        ftol: `float`. Tolerance for convergence of prediciton error
        verbose: `bool`. Whether to print progress

    Returns:

        q_m: `ndarray((nsubjects, nmodels))`. Posterior distribution over models for each subject
        a: `ndarray(nmodels)`. Posterior estimate of dirichlet parameters
        niter: `int`. Number of iterations required to converge

    Todos:

    - [ ] Notes on derivation of this inference procedure
    """
    nsubjects, nmodels = L.shape
    a_init = a.copy()

    niter = 0
    done  = False
    while not done:
        niter    += 1
        E_log_qr  = digamma(a) - digamma(a.sum())
        E_log_qr  = np.tile(E_log_qr, [nsubjects, 1])
        log_u     = L + E_log_qr
        q_m       = batch_softmax(log_u, axis=1)
        alast     = a.copy()
        beta      = q_m.sum(axis=0)
        a         = a_init + beta
        d         = np.linalg.norm(a - alast)
        if d < ftol: done = True

        if verbose:
            print('[BMS] Iteration %s | Error %s ' %(niter, d))

    return q_m, a, niter

def dirichlet_mean(alpha):
    """ Computes the mean of a Dirichlet distribution.

    Given $\mathbf p \\sim \mathrm{Dir}(\mathbf p|\mathbf \\alpha)$

    $$
    \langle p_k \\rangle = \\frac{\\alpha_k}{\sum_k \\alpha_k}
    $$

    Arguments:

        alpha: `ndarray(nmodels)`. Alpha parameters (one for each model)

    Returns:

        `ndarray(nmodels)`
    """
    return alpha/np.sum(alpha)

def dirichlet_exceedance_probability(alpha, nsamples=1000, rng=np.random.RandomState()):
    """ Exceedance probabilities for Dirichlet distribution

    Arguments:

        alpha: `ndarray(nmodels)`. Parameters of Dirichlet distribution
        nsamples: `int`. Number of samples to draw from Dirichlet
        rng: `np.random.RandomState`

    Returns:

        `ndarray(nmodels)`. Exceedance probabilities

    """
    nmodels = alpha.size
    xp = np.zeros(nmodels)
    X = dirichlet.rvs(alpha, size=nsamples, random_state=rng)
    xmax = np.argmax(X, axis=1)
    counts = np.array([xmax[xmax == i].size for i in range(nmodels)])
    return counts/np.sum(counts)

def free_energy_null(L):
    """ Computes the free energy of the null hypothesis model (Rigoux et al. 2018).

    We first introduce some notation.

    - $\mathcal H_0$: Denotes null hypothesis
    - $n_s$: number of subjects
    - $n_m$: number of models
    - $\mathbf D_i$: behavioural data for subject $i$
    - $\mathbf m_i$: one-hot vector of size $n_m$ identifying one of $n_m$ models for subject $i$
    - $\mathbf L^{\mathcal H_0} \\in \mathbb R_{-}^{n_s \\times n_m}$: matrix of log-model evidence, where $L_{ij}^{\mathcal H_0} = p(\mathbf D_i|\mathbf m_i, \mathcal H_0)$

    Rigoux et al. (2014) show that under the null hypothesis specifying

    $$
    \mathbf m_i \\sim \\prod_{j=}^{n_m} \\Big[ \\frac{1}{K} \\Big]^{m_{ij}}
    $$

    the free energy of the null model is

    $$
    \mathcal F_0 = \\sum_i\\sum_j w_{ij} (L^{\mathcal H_0}_{ij} - \\log w_{ij} - \\log n_m),
    $$

    with weights $\mathbf W = \\big\{ \{ w_{ij} \}_{j=1}^{n_m} \\big\}_{i=1}^{n_s}$ computed as

    $$
    w_{ij} = \\frac{e^{L^{\mathcal H_0}_{ij}}}{\\sum_{j'=1}^{n_m} e^{L^{\mathcal H_0}_{ij'}}
    $$

    Arguments:

        L: `ndarray((nsubjects, nmodels))`. Log-model evidence

    Returns:

        'float' Free energy of the null model

    """
    nmodels = L.shape[1]
    W = batch_softmax(L, axis=1)
    return np.sum(W*(L - np.log(nmodels) - np.ma.log(W)))

def free_energy_alternative(L, q_m, alpha_post, alpha_prior):
    """ Computes the free energy of the alternative hypothesis that models are not equally distributed in the population

    Arguments:

        alpha: `ndarray((nsubjects, nmodels))`. Parameters of the Dirichlet prior over models after convergence of the variational bayesian optimization procedure in `infer_dirichlet_multinomial_posterior`.

    Returns:

        `float`

    Todos:

    - [ ] Add notes on derivation
    """
    alpha_sum   = np.sum(alpha_post)
    E_L         = np.sum(q_m*L)
    E_log_r     = digamma(alpha_post) - digamma(alpha_sum)
    H_qr        = np.sum(gammaln(alpha_post)) - gammaln(alpha_sum) - np.sum((alpha_post-1)*E_log_r)
    H_qm        = - np.sum(q_m*np.ma.log(q_m))
    E_prior     = gammaln(alpha_prior.sum()) - np.sum(gammaln(alpha_prior)) + np.sum((alpha_prior-1)*E_log_r)
    E_log_joint = E_prior + np.sum(q_m*(E_log_r + L))
    return E_log_joint + H_qr + H_qm

def bayesian_omnibus_risk(f0, f1):
    """ Computes the Bayesian Omnibus Risk

    $$
    BOR = \\frac{1}{1+np.exp(f1-f0)}
    $$

    Arguments:

        f0: `float`. Free energy of the null model
        f1: `float`. Free energy of the alternative hypothesis model

    Returns:

        `float`
    """
    return 1/(1+np.exp(f1-f0))

def protected_exceedance_probability(xp, bor):
    """ Computes the protected exceedance probability as per Rigoux et al. (2014)

    Arguments:

        xp: `ndarray(nmodels)`. Exceedance probability
        bor: `float`. Bayesian omnibus risk

    Returns:

        `ndarray(nmodels)`. Protected exceedance probability

    """
    nmodels = xp.size
    return xp*(1-bor) + bor/nmodels
