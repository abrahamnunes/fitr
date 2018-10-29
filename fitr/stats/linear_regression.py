import numpy as np
import pandas as pd
import scipy.stats as ss

class LinearRegressionResult(object):
    def __init__(self,
                 X,
                 y,
                 coef,
                 ss_explained,
                 ss_residual,
                 ss_total,
                 C,
                 F,
                 pmodel,
                 R2adj,
                 loglik,
                 AIC,
                 BIC,
                 T,
                 coef_se,
                 pcoef,
                 intercept):
        self.intercept = intercept
        self.X = X
        self.y = y
        self.coef = coef
        self.C = C # Covariance matrix

        self.ss_explained = ss_explained
        self.ss_residual = ss_residual
        self.ss_total = ss_total

        # Hypothesis testing (Model)
        self.F = F
        self.pmodel = pmodel

        # Explained variance
        self.R2adj = R2adj

        # Loglikelihood and information criteria
        self.loglik = loglik
        self.aic = AIC
        self.bic = BIC

        # Coefficient stats
        self.tstat = T
        self.coef_se = coef_se
        self.pcoef = pcoef

    def summary(self, varnames=None, round=3):
        """ Returns a summary table

        Arguments:

            varnames: `list`. Names of the variables
        """
        if varnames is None:
            if self.intercept:
                 varnames = ['Intercept'] + ['Variable %s' %(i+1) for i in range(1, self.coef.size)]
            else:
                varnames = ['Variable %s' %(i+1) for i in range(self.coef.size)]



        elif len(varnames) < self.coef.size:
            varnames = ['Intercept'] + varnames

        df = pd.DataFrame({
            'Variable': varnames,
            'Coefficient': np.round(self.coef, round),
            'SE': np.round(self.coef_se, round),
            't': np.round(self.tstat, round),
            'p-value': np.round(self.pcoef, round)
        }, index=None)
        return df

def linear_regression(X, y, add_intercept=True, scale_x=False, scale_y=False):
    """ Performs ordinary least squares linear regression, returning MLEs of the coefficients

    ## Hypothesis testing on the model

    Compute sum of squares:

    $$
    SS_R  = (\mathbf y - \\bar{y})^\top (\mathbf y - \\bar{y})
    $$

    $$
    SS_{Res} = \mathbf y^\top \mathbf y - \mathbf w^\top \mathbf X^\top \mathbf y
    $$

    $$
    SS_T = \mathbf y^\top \mathbf y - \frac{(\mathbf 1^\top \mathbf y)^\top}{n_s}
    $$

    The test statistic is defined as follows:

    $$
    F = \frac{SS_R (n-k-1)}{SS_{Res} k} \sim F(k, n-k-1)
    $$

    The adjusted $R^2$ is

    $$
    R^2_{Adj} = 1 - \frac{SS_R (n-1)}{SS_T (n-k-1)}
    $$

    ## Hypothesis testing on the coefficients

    The test statistic is

    $$
    \frac{w_i}{SE(w_i)} \sim StudentT(n-k-1)
    $$


    Arguments:

        X: `ndarray((nsamples, nfeatures))`. Predictors
        y: `ndarray(nsamples)`. Target
        add_intercept: `bool`. Whether to add an intercept term (pads on LHS of `X` with column of ones)
        scale_x: `bool`. Whether to scale the columns of `X`
        scale_y: `bool`. Whether to scale the columns of `y`

    Returns:

        `LinearRegressionResult`
    """
    nsamples, nfeatures = X.shape
    dfd = nfeatures
    dfn = nsamples-nfeatures-1

    if add_intercept:
        X = np.hstack((np.ones((X.shape[0], 1)), X))

    if scale_x:
        X = fu.scale_data(X)

    if scale_y:
        y = np.ravel(fu.scale_data(y))

    # Compute coefficients
    XTXinv = np.linalg.pinv(X.T@X)
    coef = XTXinv@(X.T@y)

    # Compute sum of squares
    ybar = np.mean(y)
    yhat = X@coef
    ss_explained = (yhat-ybar).T@(yhat-ybar)
    ss_residual = y.T@y - coef.T@(X.T@y)
    ss_total = y.T@y - np.square(np.sum(y))/nsamples
    variance = ss_residual/nsamples
    C = variance*XTXinv

    # Hypothesis test on the model
    F = (ss_explained * dfn)/(ss_residual * dfd)
    pmodel = 1 - ss.f.cdf(F, dfd, dfn)

    # Compute explained variance
    R2adj = 1 - (ss_explained * (nsamples - 1))/(ss_total * dfn)
    loglik = -(1/2)*(np.log(2*np.pi*variance) + (ss_residual/variance))
    AIC = 2*nfeatures - 2*loglik
    BIC = nfeatures*np.log(nsamples) - 2*loglik

    # Compute standard errors and p-values for coefficients
    coef_se = np.sqrt(np.diag(C))
    T = coef/coef_se
    pcoef = 2*(1 - ss.t(dfn).cdf(np.abs(T)))

    res = LinearRegressionResult(
        X=X,
        y=y,
        coef=coef,
        ss_explained=ss_explained,
        ss_residual=ss_residual,
        ss_total=ss_total,
        C=C,
        F=F,
        pmodel=pmodel,
        R2adj=R2adj,
        loglik=loglik,
        AIC=AIC,
        BIC=BIC,
        T=T,
        coef_se=coef_se,
        pcoef=pcoef,
        intercept=add_intercept
    )
    return res
