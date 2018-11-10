# `fitr.stats`

Functions for statistical analyses. 



## bic

```python
fitr.stats.model_evaluation.bic(log_prob, nparams, ntrials)
```

Bayesian Information Criterion (BIC)

Arguments:

- **log_prob**: Log probability
- **nparams**: Number of parameters in the model
- **ntrials**: Number of trials in the time series

Returns:

Scalar estimate of BIC.

---



## lme

```python
fitr.stats.model_evaluation.lme(log_prob, nparams, hess_inv)
```

Laplace approximation to the log model evidence

Arguments:

- **log_prob**: Log probability
- **nparams**: Number of parameters in the model
- **hess_inv**: Hessian at the optimum (shape is $K \times K$)

Returns:

Scalar approximation of the log model evidence

---



## pearson_rho

```python
fitr.stats.correlations.pearson_rho(X, Y, comparison='diagonal')
```

Linear (Pearson) correlation coefficient.

Will compute the following formula

$$
\rho = \frac{\mathbf x^\top \mathbf y}{\lVert \mathbf x Vert \cdot \lVert \mathbf y Vert}
$$

where each vector $\mathbf x$ and $\mathbf y$ are rows of the matrices $\mathbf X$ and $\mathbf Y$, respectively.

Also returns a two-tailed p-value where the hypotheses being tested are

$$
H_o: \rho = 0
$$

$$
H_a: \rho \neq 0
$$

and where the test statistic is

$$
T = \frac{\rho \sqrt{n_s-2}}{\sqrt{1 - \rho^2}}
$$

and the p-value is thus

$$
p = 2*(1 - \mathcal T(T, n_s-2))
$$

given the CDF of the Student T-distribution with degrees of freedom $n_s-2$.

Arguments:

- **X**: `ndarray((nsamples, nfeatures))` of dimension 1 or 2. If `X` is a 1D array, it will be converted to 2D prior to computation
- **Y**: `ndarray((nsamples, nfeatures))` of dimension 1 or 2. If `Y` is a 1D array, it will be converted to 2D prior to computation
- **comparison**: `str`. Here `'diagonal'` computes correlations individually, column-for-column between matrices. Otherwise `'pairwise'` computes pairwise correlations between columns in `X` and `Y`.

Returns:

- **rho**: `ndarray((nfeatures,))`. Correlation coefficient(s). Will be an `X.shape[1]` by `Y.shape[1]` matrix if `comparison='pairwise'`
- **p**: `ndarray((nfeatures,))`. P-values for correlation coefficient(s). Will be an `X.shape[1]` by `Y.shape[1]` matrix if `comparison='pairwise'`



TODO:

- [ ] Create error raised when X and Y are not same dimension

---



## spearman_rho

```python
fitr.stats.correlations.spearman_rho(X, Y, comparison='diagonal')
```

Spearman's rank correlation 

Note this function takes correlations between the columns of `X` and `Y`. 

Arguments:

- **X**: `ndarray((nsamples, nfeatures))` of dimension 1 or 2. If `X` is a 1D array, it will be converted to 2D prior to computation
- **Y**: `ndarray((nsamples, nfeatures))` of dimension 1 or 2. If `Y` is a 1D array, it will be converted to 2D prior to computation
- **comparison**: `str`. Here `'diagonal'` computes correlations individually, column-for-column between matrices. Otherwise `'pairwise'` computes pairwise correlations between columns in `X` and `Y`.

Returns:

- **rho**: `ndarray((nfeatures,))`. Correlation coefficient(s). Will be an `X.shape[1]` by `Y.shape[1]` matrix if `comparison='pairwise'`
- **p**: `ndarray((nfeatures,))`. P-values for correlation coefficient(s). Will be an `X.shape[1]` by `Y.shape[1]` matrix if `comparison='pairwise'`

    

---



## linear_regression

```python
fitr.stats.linear_regression.linear_regression(X, y, add_intercept=True, scale_x=False, scale_y=False)
```

Performs ordinary least squares linear regression, returning MLEs of the coefficients

## Hypothesis testing on the model

Compute sum of squares:

$$
SS_R  = (\mathbf y - \bar{y})^      op (\mathbf y - \bar{y})
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

- **X**: `ndarray((nsamples, nfeatures))`. Predictors
- **y**: `ndarray(nsamples)`. Target
- **add_intercept**: `bool`. Whether to add an intercept term (pads on LHS of `X` with column of ones)
- **scale_x**: `bool`. Whether to scale the columns of `X`
- **scale_y**: `bool`. Whether to scale the columns of `y`

Returns:

`LinearRegressionResult`

---



## kruskal_wallis

```python
fitr.stats.nonparametric.kruskal_wallis(x, g, dist='beta')
```

Kruskal-Wallis one-way analysis of variance (one-way ANOVA on ranks)

Arguments:

- **x**: `ndarray(nsamples)`. Vector of data to be compared
- **g**: `ndarray(nsamples)`. Group ID's
- **dist**: `str {'chi2', 'beta'}`. Which distributional approximation to make

Returns:

- **T**: `float`. Test statistic
- **p**: `float`. P-value for the comparison

---



## conover

```python
fitr.stats.nonparametric.conover(x, g, alpha=0.05, adjust='bonferroni')
```

Conover's nonparametric test of homogeneity.

Arguments:

- **x**: `ndarray(nsamples)`. Vector of data to be compared
- **g**: `ndarray(nsamples)`. Group ID's
- **alpha**: `0 < float < 1`. Significance threshold
- **adjust**: `str`. Method to adjust p-values (see below)

Returns:

- **T**: `float`. Test statistic
- **p**: `float`. P-value for the comparison

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

---



## confusion_matrix

```python
fitr.stats.confusion_matrix.confusion_matrix(ytrue, ypred)
```

Creates a confusion matrix from some ground truth labels and predictions 

Arguments: 

- **ytrue**: `ndarray(nsamples)`. Ground truth labels 
- **ypred**: `ndarray(nsamples)`. Predicted labels 

Returns: 

- **C**: `ndarray((nlabels, nlabels))`. Confusion matrix 


Example: 

In the binary classification case, we may have the following: 

``` python 
from fitr.stats import confusion_matrix

ytrue = np.array([0, 1, 0, 1, 0])
ypred = np.array([1, 1, 0, 1, 0])
C = confusion_matrix(ytrue, ypred)
tn, fp, fn, tp = C.flatten()
```

---


