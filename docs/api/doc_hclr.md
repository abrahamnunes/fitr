# `fitr.hclr`

Hierarchical convolutional logistic regression (HCLR): A general analysis method for trial-by-trial behavioural data with covariates.



## HCLR

```python
fitr.hclr.HCLR()
```

Hierarchical Convolutional Logistic Regression (HCLR) for general behavioural data.


Attributes:

- **X**: `ndarray((nsubjects,ntrials,nfeatures))`. The ``experience'' tensor.
- **y**: `ndarray((nsubjects,ntrials,ntargets))`. Tensor of ``choices'' we are trying to predict.
- **Z**: `ndarray((nsubjects,ncovariates))`. Covariates of interest
- **V**: `ndarray((naxes,nfeatures))`. Vectors identifying features of interest (i.e. to compute indices). If `add_intercept=True`, then the dimensionality of `V` should be `ndarray((naxes, nfeatures+1))`, where the first column represents the basis coordinate for the bias.
- **filter_size**: `int`. Number of steps prior to target included as features.
- **loading_matrix_scale**: `float > 0`. Scale of the loading matrix $\boldsymbol\Phi$, which is assumed that $\phi_{ij} \sim \mathcal N(0, 1)$, with the default scale being 1.
- **add_intercept**: `bool'. Whether to add intercept
- **group_mean**: `ndarray`. Samples of the posterior group-level mean. `None` until model is fit
- **group_scale**: `ndarray`. Samples of the posterior group-level scale. `None` until model is fit
- **loading_matrix**: `ndarray`. Samples of the posterior loading matrix. `None` until model is fit
- **subject_parameters**: `ndarray`. Samples of the posterior subject-level parameters. `None` until model is fit
- **group_indices**: `ndarray`. Samples of the posterior group-level projections on to the basis. `None` until model is fit
- **covariate_effects**: `ndarray`. Samples of the posterior projection of the loading matrix onto the basis. `None` until model is fit

## Notes

- When presenting `X` and `y`, note that the indices of `y` should correspond exactly to the trial indices in `X`, even though the HCLR analysis is predicting a trial ahead. In other words, there should be no lag in the `X`, `y` inputs. The HCLR setup will automatically set up the lag depending on how you set the `filter_size`.

---




### HCLR.fit

```python
fitr.hclr.fit(self, nchains=4, niter=1000, warmup=None, thin=1, seed=None, verbose=False, algorithm='NUTS', n_jobs=-1)
```

Fits the HCLR model

Arguments:

- **nchains**: `int`. Number of chains for the MCMC run.
- **niter**: `int`. Number of iterations over which to run MCMC.
- **warmup**: `int`. Number of warmup iterations
- **thin**:  `int`. Periodicity of sample recording
- **seed**: `int`. Seed for pseudorandom number generator
- **algorithm**: `{'NUTS','HMC'}`
- **n_jobs**: `int`. Number of cores to use (default=-1, as many as possible and required)

---


