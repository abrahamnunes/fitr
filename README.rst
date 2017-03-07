fitr |Build status| |Documentation Status|
==========================================

Python implementation of package to fit reinforcement learning models to
behavioural data

Tutorial
--------

Import package components
~~~~~~~~~~~~~~~~~~~~~~~~~

-  Assumes your current directory is the ``fitr`` folder

Generate some synthetic data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    import numpy as np
    from rlparams import *
    from tasks import bandit

    nsubjects=50

    # Initialize array of parameter values [learning rate, choice randomness]
    params = np.zeros([nsubjects, 2])

    # Sample parameter values from rlparams objects
    params[:,0] = LearningRate().dist.rvs(size=nsubjects)
    params[:,1] = ChoiceRandomness().dist.rvs(size=nsubjects)

    # Run task
    res = bandit().simulate(nsubjects=nsubjects,
                            ntrials=100,
                            params=params)

    # Plot the cumulative reward
    res.plot_cumreward()

    # Scatterplots of total reward vs parameter values
    res.cumreward_param_plot()

Fit a reinforcement learning model to the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from fitr import *
    from loglik_functions import bandit_ll

    # Model with learning rate and choice randomness
    lrcr_model = fitrmodel(loglik_func=bandit_ll().lr_cr,
                           params=[LearningRate(),ChoiceRandomness()])

Now that the models are created, we can fit them. The default method
(the only one presently implemented) is Expectation-Maximization.

.. code:: python

    lrcr_fit   = lrcr_model.fit(data=res.data)

Then you can plot the actual vs. estimated parameters as follows:

.. code:: python

    lrcr_fit.plot_ae(actual=res.params)

Or you can also plot histograms of the parameter estimates:

.. code:: python

    lrcr_fit.param_hist()

You can also plot the progression in Log-Model-Evidence, BIC, and AIC
(whole model, not subject level) over the course of model fitting. LME
should increase and then plateau, whereas BIC and AIC should decrease,
then plateau. If there are deviations in the opposite direction for any
of those, model fitting can be run again.

.. code:: python

    lrcr_fit.plot_fit_ts()

.. |Build status| image:: 
   :target: https://travis-ci.org/ComputationalPsychiatry/fitr
.. |Documentation Status| image:: https://readthedocs.com/projects/computationalpsychiatry-fitr/badge/?version=latest
   :target: https://computationalpsychiatry-fitr.readthedocs-hosted.com/en/latest/?badge=latest