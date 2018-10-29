import numpy as np
import scipy.stats as ss
from statsmodels.stats.multitest import multipletests

def rank_grouped_data(x, g):
    """ Ranks observations taken across several groups

    Arguments:

        x: `ndarray(nsamples)`. Vector of data to be compared
        g: `ndarray(nsamples)`. Group ID's

    Returns:

        counts: `ndarray(n_unique_observations)`. Number of observations for each unique value of the observations
        ranks: `ndarray(nsamples)`. Ranks for each observation
        G: `ndarray(nsamples, ngroups)`.  Matrix indicating whether sample i is in group j
        R: `ndarray((nsamples, ngroups))`. Matrix indicating the rank for sample i in group j
        lab: `ndarray(ngroups)`. Group labels
    """
    nsamples = x.size
    ngroups = np.unique(g).size
    ranks = np.empty(nsamples)

    # Sort in ascending order
    idx   = np.argsort(x)
    G,lab = make_onehot(g[idx])

    # Compute number of each value
    counts, bins = np.histogram(x, bins=np.unique(x))

    # Assign ranks, with ties getting average of the ranks they would have
    #   achieved without being tied
    for i, val in enumerate(bins[:-1]):
        if i == 0:
            ranks[:counts[i]] = np.mean(np.arange(counts[i])+1)
        else:
            start = np.sum(counts[:i])
            end = np.sum(counts[:i+1])
            value_ranks = np.arange(start, end) + 1
            ranks[start:end] = np.mean(value_ranks)

    R = np.tile(ranks.reshape(-1, 1), [1, ngroups]) * G
    return counts, ranks, G, R, lab

def kruskal_wallis(x, g, dist='beta'):
    """ Kruskal-Wallis one-way analysis of variance (one-way ANOVA on ranks)

    Arguments:

        x: `ndarray(nsamples)`. Vector of data to be compared
        g: `ndarray(nsamples)`. Group ID's
        dist: `str {'chi2', 'beta'}`. Which distributional approximation to make

    Returns:

        T: `float`. Test statistic
        p: `float`. P-value for the comparison

    """
    nsamples = x.size
    ngroups = np.unique(g).size

    counts, ranks, G, R, lab = rank_grouped_data(x, g)

    # Compute test statistic        uniqueranks
    n_per_group = np.sum(G, axis=0)
    ranksum = np.sum(R, axis=0)
    meanranksumsq = np.sum(ranksum**2/n_per_group)
    T = (12/(nsamples*(nsamples+1)))*meanranksumsq - 3*(nsamples+1)

    # Correct for ties
    if np.any(np.greater(counts, 1)):
        C = 1 - np.sum(counts**3 - counts)/(nsamples**3 - nsamples)
        T = T/C

    # Compute (approximate) p-value
    if dist == 'chi2':
        p = 1 - ss.chi2(df=ngroups-1).cdf(T)

    elif dist=='beta':
        # Expected value of T
        mu = ngroups - 1

        # Variance of T
        a = 2*(ngroups-1)
        b = 2*(3*ngroups**2 - 6*ngroups + nsamples*(2*ngroups**2 - 6*ngroups + 1))
        c = 5*nsamples*(nsamples+1)
        d = (6/5)*np.sum(np.ones(ngroups)/n_per_group)
        var = a - (b/c) - d

        # Maximum value of T
        eta = (nsamples**3 - np.sum(n_per_group**3))/(nsamples*(nsamples+1))

        # Compute beta parameters
        shape_a = mu*((mu*(eta-mu) - var)/(eta*var))
        shape_b = shape_a*((eta-mu)/mu)

        p = 1 - ss.beta.cdf(T/eta, shape_a, shape_b)

    return T, p

def conover(x, g, alpha=0.05, adjust='bonferroni'):
    """ Conover's nonparametric test of homogeneity.

    Arguments:

        x: `ndarray(nsamples)`. Vector of data to be compared
        g: `ndarray(nsamples)`. Group ID's
        alpha: `0 < float < 1`. Significance threshold
        adjust: `str`. Method to adjust p-values (see below)

    Returns:

        T: `float`. Test statistic
        p: `float`. P-value for the comparison

    Notes:

        Adjustment methods include the following:

        - `bonferroni` : one-step correction
        - `sidak` : one-step correction
        - `holm-sidak` : step down method using Sidak adjustments
        - `holm` : step-down method using Bonferroni adjustments
        - `simes-hochberg` : step-up method  (independent)
        - `hommel` : closed method based on Simes tests (non-negative)
        - `fdr_bh` : Benjamini/Hochberg  (non-negative)
        - `fdr_by` : Benjamini/Yekutieli (negative)
        - `fdr_tsbh` : two stage fdr correction (non-negative)
        - `fdr_tsbky` : two stage fdr correction (non-negative)

    References:

        W. J. Conover and R. L. Iman (1979), On multiple-comparisons procedures, Tech. Rep. LA-7677-MS, Los Alamos Scientific Laboratory.

    """
    counts, ranks, G, R, lab = rank_grouped_data(x, g)
    nsamples, ngroups = G.shape
    n_per_group = np.sum(G, axis=0)
    ranksum = np.sum(R, axis=0)

    # Compute absolute rank-sum differences and inverse sample size ratios
    RSD = np.empty((ngroups, ngroups))
    ISR = np.empty((ngroups, ngroups))
    LAB = np.empty((ngroups, ngroups), dtype='U100')
    for i in range(ngroups):
        for j in range(ngroups):
            rsr_i = ranksum[i]/n_per_group[i]
            rsr_j = ranksum[j]/n_per_group[j]
            RSD[i,j] = np.abs(rsr_i-rsr_j)

            isr_i = 1/n_per_group[i]
            isr_j = 1/n_per_group[j]
            ISR[i, j] = np.sqrt(isr_i + isr_j)

            LAB[i, j] = lab[i] + '-' + lab[j]

    # Compute S2
    a     = 1/(nsamples-1)
    sumR2 = np.sum(R**2)
    b     = nsamples*((nsamples+1)**2/4)
    S2    = a*(sumR2 - b)

    # Compute Kuskal-Wallis test statistic
    T = (1/S2)*(np.sum(ranksum**2/n_per_group) - ((nsamples*(nsamples+1)**2)/4))

    # Compute conover stat
    c = np.sqrt(S2*((nsamples-1-T)/(nsamples-ngroups)))
    t_stat = RSD/(c*ISR)

    p = 2 * ss.t(df=nsamples-ngroups).sf(t_stat)

    # Create table prior to adjustment
    msk = np.equal(np.tril(np.ones((ngroups, ngroups)), k=-1), 1)
    lab = LAB[msk]
    p   = p[msk]

    if adjust is not None:
        reject, p, _, _ = multipletests(pvals=p,
                                        method=adjust,
                                        is_sorted=False,
                                        returnsorted=False)

    return p, reject, lab
