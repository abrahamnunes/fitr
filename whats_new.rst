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

- ``twostep`` module adds various models for the two-step task.

Enhancements
------------

- Reinforcement learning parameters in module ``rlparams`` can now be initialized using desired mean and standard deviations. This should allow easier simulation of synthetic data from various tasks.
- The ``SyntheticData`` object was moved to a new module, ``fitrdata`` to account for the new task-specific model structures being introduced.

Bug Fixes
---------

Deprecations
------------

- The modules ``tasks``, ``loglik_functions``, and ``generative_models`` are deprecated and will be replaced by task specific modules, such as ``twostep``, wherein each model of the task will be represented as an object with the relevant likelihood functions and generative models contained within.

Removed Features
----------------

- ``rlparams.generate_groups()``

Version 0.0.1
=============

**April 8, 2017**

New Features
------------

- Model fitting with Markov-Chain Monte Carlo (via Stan)
