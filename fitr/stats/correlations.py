import numpy as np
import scipy.stats as ss
from fitr.utils import scale_data

def pearson_rho(X, Y):
    """ Linear (Pearson) correlation coefficient.

    Will compute the following formula

    $$
    \\rho = \\frac{\mathbf x^\\top \mathbf y}{\lVert \mathbf x \rVert \cdot \lVert \mathbf y \rVert}
    $$

    where each vector $\mathbf x$ and $\mathbf y$ are rows of the matrices $\mathbf X$ and $\mathbf Y$, respectively.

    Also returns a two-tailed p-value where the hypotheses being tested are

    $$
    H_o: \\rho = 0
    $$

    $$
    H_a: \\rho \\neq 0
    $$

    and where the test statistic is

    $$
    T = \\frac{\\rho \\sqrt{n_s-2}}{\\sqrt{1 - \\rho^2}}
    $$

    and the p-value is thus

    $$
    p = 2*(1 - \\mathcal T(T, n_s-2))
    $$

    given the CDF of the Student T-distribution with degrees of freedom $n_s-2$.

    Arguments:

        X: `ndarray((nsamples, nfeatures))` of dimension 1 or 2. If `X` is a 1D array, it will be converted to 2D prior to computation
        Y: `ndarray((nsamples, nfeatures))` of dimension 1 or 2. If `Y` is a 1D array, it will be converted to 2D prior to computation

    Returns:

        rho: `ndarray((nfeatures,))`. Correlation coefficient(s)

    TODO:

    - [ ] Create error raised when X and Y are not same dimension
    """
    # Reshape if necessary
    if X.ndim == 1 and Y.ndim == 1:
        X = X.reshape(-1, 1) - np.mean(X)
        Y = Y.reshape(-1, 1) - np.mean(Y)

    X = scale_data(X, axis=0, with_mean=True, with_var=False)
    Y = scale_data(Y, axis=0, with_mean=True, with_var=False)

    xnorm = np.linalg.norm(X, axis=0, ord=2)
    ynorm = np.linalg.norm(Y, axis=0, ord=2)
    rho = np.diag(X.T@Y)/(xnorm*ynorm)


    # Compute the p-values of the correlation
    n  = X.shape[0]
    df = n - 2
    T  = (rho*np.sqrt(df))/np.sqrt(1-rho**2)
    p  = 2*(1 - ss.t.cdf(np.abs(T), df=df))

    return rho, p
