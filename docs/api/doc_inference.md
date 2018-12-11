# `fitr.inference`

Methods for inferring the parameters of generative models for reinforcement learning data.



## OptimizationResult

```python
fitr.inference.optimization_result.OptimizationResult()
```

Container for the results of an optimization run on a generative model of behavioural data

Arguments:

- **subject_id**: `ndarray((nsubjects,))` or `None` (default). Integer ids for subjects
- **xmin**: `ndarray((nsubjects,nparams))` or `None` (default). Parameters that minimize objective function
- **fmin**: `ndarray((nsubjects,))` or `None` (default). Value of objective function at minimum
- **fevals**: `ndarray((nsubjects,))` or `None` (default). Number of function evaluations required to minimize objective function
- **niters**: `ndarray((nsubjects,))` or `None` (default). Number of iterations required to minimize objective function
- **lme**: `ndarray((nsubjects,))` or `None` (default). Log model evidence
- **bic**: `ndarray((nsubjects,))` or `None` (default). Bayesian Information Criterion
- **hess_inv**: `ndarray((nsubjects,nparams,nparams))` or `None` (default). Inverse Hessian at the optimum.
- **err**: `ndarray((nsubjects,nparams))` or `None` (default). Error of estimates at optimum.

---




### OptimizationResult.transform_xmin

```python
fitr.inference.optimization_result.transform_xmin(self, transforms, inplace=False)
```

Rescales the parameter estimates.

Arguments:

- **transforms**: `list`. Transformation functions where `len(transforms) == self.xmin.shape[1]`
- **inplace**: `bool`. Whether to change the values in `self.xmin`. Default is `False`, which returns an `ndarray((nsubjects, nparams))` of the transformed parameters.

Returns:

`ndarray((nsubjects, nparams))` of the transformed parameters if `inplace=False`

---



## mlepar

```python
fitr.inference.mle_parallel.mlepar(f, data, nparams, minstarts=2, maxstarts=10, maxstarts_without_improvement=3, init_sd=2, njobs=-1, jac=None, hess=None, method='L-BFGS-B')
```

Computes maximum likelihood estimates using parallel CPU resources.

Wraps over the `fitr.optimization.mle_parallel.mle` function.

Arguments:

- **f**: Likelihood function
- **data**: A subscriptable object whose first dimension indexes subjects
- **optimizer**: Optimization function (currently only `l_bfgs_b` supported)
- **nparams**: `int` number of parameters to be estimated
- **minstarts**: `int`. Minimum number of restarts with new initial values
- **maxstarts**: `int`. Maximum number of restarts with new initial values
- **maxstarts_without_improvement**: `int`. Maximum number of restarts without improvement in objective function value
- **init_sd**: Standard deviation for Gaussian initial values
- **jac**: `bool`. Set to `True` if `f` returns a Jacobian as the second element of the returned values
- **hess**: `bool`. Set to `True` if third output value of `f` is the Hessian matrix
- **method**: `str`. One of the `scipy.optimize` methods.

Returns:

`fitr.inference.OptimizationResult`

Todo:

- [ ] Raise errors when user selects inappropriate optimization function given values for `jac` and `hess`

---



## l_bfgs_b

```python
fitr.inference.mle_parallel.l_bfgs_b(f, i, data, nparams, jac, minstarts=2, maxstarts=10, maxstarts_without_improvement=3, init_sd=2)
```

Minimizes the negative log-probability of data with respect to some parameters under function `f` using the L-BFGS-B algorithm.

This function is specified for use with parallel CPU resources.

Arguments:

- **f**: (Negative!) Log likelihood function
- **i**: `int`. Subject being optimized (slices first dimension of `data`)
- **data**: Object subscriptable along first dimension to indicate subject being optimized
- **nparams**: `int`. Number of parameters in the model
- **jac**: `bool`. Set to `True` if `f` returns a Jacobian as the second element of the returned values
- **minstarts**: `int`. Minimum number of restarts with new initial values
- **maxstarts**: `int`. Maximum number of restarts with new initial values
- **maxstarts_without_improvement**: `int`. Maximum number of restarts without improvement in objective function value
- **init_sd**: Standard deviation for Gaussian initial values

Returns:

- **i**: `int`. Subject being optimized (slices first dimension of `data`)
- **xmin**: `ndarray((nparams,))`. Parameter values at optimum
- **fmin**: Scalar objective function value at optimum
- **fevals**: `int`. Number of function evaluations
- **niters**: `int`. Number of iterations
- **lme_**: Scalar log-model evidence at optimum
- **bic_**: Scalar Bayesian Information Criterion at optimum
- **hess_inv**: `ndarray((nparams, nparams))`. Inv at optimum

---



## bms

```python
fitr.inference.bms.bms(L, ftol=1e-12, nsamples=1000000, rng=<mtrand.RandomState object at 0x7f811aaddfc0>, verbose=True)
```

Implements variational Bayesian Model Selection as per Rigoux et al. (2014).

Arguments:

- **L**: `ndarray((nsubjects, nmodels))`. Log model evidence
- **ftol**: `float`. Threshold for convergence of prediction error
- **nsamples**: `int>0`. Number of samples to draw from Dirichlet distribution for computation of exceedence probabilities
- **rng**: `np.random.RandomState`
- **verbose**: `bool (default=True)`. If `False`, no output provided.

Returns:

- **pxp**: `ndarray(nmodels)`. Protected exceedance probabilities
- **xp**: `ndarray(nmodels)`. Exceedance probabilities
- **bor**: `ndarray(nmodels)`. Bayesian Omnibus Risk
- **q_m**: `ndarray((nsubjects, nmodels))`. Posterior distribution over models for each subject
- **alpha**: `ndarray(nmodels)`. Posterior estimates of Dirichlet parameters
- **f0**: `float`. Free energy of null model
- **f1**: `float`. Free energy of alternative model
- **niter**: `int`. Number of iterations of posterior optimization

Examples:

Assuming one is given a matrix of (log-) model evidence values `L` of type `ndarray((nsubjects, nmodels))`,

```
from fitr.inference import spm_bms

pxp, xp, bor, q_m, alpha, f0, f1, niter = bms(L)
```

Todos:

- [ ] Add notes on derivation

---


