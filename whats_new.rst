.. -*- mode: rst -*-

====================
Fitr Release History
====================

Version 0.0.2
=============

**In Development**

As of this version, development and testing will be done exclusively in Python 3.5+.

New Features
------------

Summary
.......

- ``inference``, ``metrics``, ``models``, ``model_selection``, ``plotting``, ``unsupervised`` subpackages
- ``twostep`` module adds various models for the two-step task. Replaces ``tasks.twostep`` class
- Reinforcement learning parameters in module ``rlparams`` can now be initialized using desired mean and standard deviations. This should allow easier simulation of synthetic data from various tasks.
- Added ``rlparams.Param.plot_pdf`` function to plot the probability density function of synthetic parameters

Inference Subpackage
........................

- The former ``fitr`` module has been turned into the subpackage ``inference``, with all constituent functions therein. Should make development a bit easier with shorter files. Also importing ``fitr.inference`` makes a bit more sense than ``fitr.fitr``, as it was before.
- Implemented ``fitr.inference.MLE()`` class for maximum-likelihood estimation

Metrics Subpackage
..................

- Model evaluation functions like ``BIC``, ``AIC``, and ``LME`` are now here
- A new ``distance`` module which contains distance metrics
    - ``parameter_distance``
    - ``likelihood_distance``

Models Subpackage
.................

- A new place to keep all of the paradigm model modules
- ``twostep`` module containing models of the two-step task
- ``synthetic_data`` module containing object for synthetic behavioural data

Model Selection Subpackage
..........................

- The former ``model_selection`` module is now the ``fitr.model_selection`` subpackage, with constituent functions as individual modules. Should make development easier with shorter files.

Plotting Subpackage
...................

- A new place to write the plotting functions to be used across Fitr
- ``heatmap`` function
- ``distance_hist`` and ``distance_scatter`` functions

Unsupervised Subpackage
.......................

- ``cluster`` module including ``AffinityPropagation`` algorithm
- ``embedding`` module including ``TSNE`` algorithm

Enhancements
------------

- Added more unit tests to catch up on code coverage

Bug Fixes
---------

- Fixed problem with automatic sizing of ``fitrfit.plot_ae()``

Deprecations
------------

- The modules ``tasks``, ``loglik_functions``, and ``generative_models`` are deprecated and will be replaced by task specific modules, such as ``twostep``, wherein each model of the task will be represented as an object with the relevant likelihood functions and generative models contained within.

Removed Features
----------------

- ``tasks.twostep``
- ``rlparams.generate_groups``
- 'MAP0' option in the ``fitr.inference.fitmodel`` function ``FitModel.fit()``

Version 0.0.1
=============

**April 8, 2017**

New Features
------------

- Model fitting with Markov-Chain Monte Carlo (via Stan)

List of Contributors
====================

- Abraham Nunes (Dalhousie University. Halifax, NS, Canada)
- Alexander Rudiuk (Dalhousie University. Halifax, NS, Canada)
