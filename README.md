#  fitr [![Build status](https://ci.appveyor.com/api/projects/status/2cbutmj6i890uujj?svg=true)](https://ci.appveyor.com/project/abrahamnunes/fitr)  [![Documentation Status](https://readthedocs.com/projects/computationalpsychiatry-fitr/badge/?version=latest)](https://computationalpsychiatry-fitr.readthedocs-hosted.com/en/latest/?badge=latest)

Python implementation of package to fit reinforcement learning models to behavioural data


## Tutorial

### Generate some synthetic data

``` python
import numpy as np
import fitr
from fitr import tasks, loglik_functions, model_selection


nsubjects=50

# Initialize array of parameter values [learning rate, choice randomness]
params = np.zeros([nsubjects, 2])

# Sample parameter values from rlparams objects
params[:,0] = fitr.LearningRate().dist.rvs(size=nsubjects)
params[:,1] = fitr.ChoiceRandomness().dist.rvs(size=nsubjects)

# Run task
res = tasks.bandit().simulate(nsubjects=nsubjects,
                              ntrials=100,
                              params=params)

# Plot the cumulative reward
res.plot_cumreward()

# Scatterplots of total reward vs parameter values
res.cumreward_param_plot()

```

### Fit a reinforcement learning model to the data

``` python

# Model with learning rate and choice randomness
lrcr_model = fitrmodel(loglik_func=loglik_functions.bandit_ll().lr_cr,
                       params=[fitr.LearningRate(),fitr.ChoiceRandomness()])
```

Now that the models are created, we can fit them. The default method (the only one presently implemented) is Expectation-Maximization.

``` python
lrcr_fit   = lrcr_model.fit(data=res.data)
```

Then you can plot the actual vs. estimated parameters as follows:

``` python
lrcr_fit.plot_ae(actual=res.params)
```

Or you can also plot histograms of the parameter estimates:

``` python
lrcr_fit.param_hist()
```

You can also plot the progression in Log-Model-Evidence, BIC, and AIC (whole model, not subject level) over the course of model fitting. LME should increase and then plateau, whereas BIC and AIC should decrease, then plateau. If there are deviations in the opposite direction for any of those, model fitting can be run again.

``` python
lrcr_fit.plot_fit_ts()
```
