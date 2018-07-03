# `fitr.metrics`

Metrics and performance statistics.



## bic

```python
fitr.metrics.bic(log_prob, nparams, ntrials)
```

Bayesian Information Criterion (BIC)

Arguments:

- **log_prob**: Log probability
- **nparams**: Number of parameters in the model
- **ntrials**: Number of trials in the time series

Returns:

Scalar estimate of BIC.

---



## linear_correlation

```python
fitr.metrics.linear_correlation(X, Y)
```

Linear correlation coefficient.

Will compute the following formula

$$
\rho = \frac{\mathbf x^\top \mathbf y}{\lVert \mathbf x Vert \cdot \lVert \mathbf y Vert}
$$

where each vector $\mathbf x$ and $\mathbf y$ are rows of the matrices $\mathbf X$ and $\mathbf Y$, respectively.

Arguments:

- **X**: `ndarray((nsamples, nfeatures))` of dimension 1 or 2. If `X` is a 1D array, it will be converted to 2D prior to computation
- **Y**: `ndarray((nsamples, nfeatures))` of dimension 1 or 2. If `Y` is a 1D array, it will be converted to 2D prior to computation

Returns:

- **rho**: `ndarray((nfeatures,))`. Correlation coefficient(s)

TODO:

- [ ] Create error raised when X and Y are not same dimension

---



## lme

```python
fitr.metrics.lme(log_prob, nparams, hess_inv)
```

Laplace approximation to the log model evidence

Arguments:

- **log_prob**: Log probability
- **nparams**: Number of parameters in the model
- **hess_inv**: Hessian at the optimum (shape is $K \times K$)

Returns:

Scalar approximation of the log model evidence

---



## log_loss

```python
fitr.metrics.log_loss(p, q)
```

Computes log loss.

$$
\mathcal L = \mathbf p^\top \log \mathbf q + (1-\mathbf p)^\top \log (1 - \mathbf q)
$$

Arguments:

- **p**: Binary vector of true labels `ndarray((nsamples,))`
- **q**: Vector of estimates (between 0 and 1) of type `ndarray((nsamples,))`

Returns:

Scalar log loss

---


