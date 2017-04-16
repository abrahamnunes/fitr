Fitting Models to Data
======================

Model-Fitting Methods in Fitr
-----------------------------

Fitr implements several model-fitting methods:

=============================== ========================== ===============
Method                          Function                   Reference
=============================== ========================== ===============
EM with Laplace Approximation   ``fitr.EM()``              [Huys2011]_
Empirical Priors                ``fitr.EmpiricalPriors()`` [Gershman2016]_
Markov Chain Monte-Carlo (Stan) ``fitr.MCMC()``            [StanDevs]_
=============================== ========================== ===============

Here, "EM" refers to Expectation-Maximization.

References
----------
.. [Gershman2016] Gershman, S.J. (2016) Empirical priors for reinforcement learning models. J. Math. Psychol. 71, 1–6
.. [Huys2011] Huys, Q. J. M., et al. (2011). Disentangling the roles of approach, activation and valence in instrumental and pavlovian responding. PLoS Computational Biology, 7(4).
.. [StanDevs] Stan Development Team. 2016. PyStan: the Python interface to Stan, Version 2.14.0.0.   http://mc-stan.org
