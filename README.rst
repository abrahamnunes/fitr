.. -*- mode: rst -*-

|PyPI|_ |Build|_ |Health|_ |Codecov|_ |PyV|_ |DOI|_

.. |Build| image:: https://travis-ci.org/ComputationalPsychiatry/fitr.svg?branch=master
.. _Build: https://travis-ci.org/ComputationalPsychiatry/fitr

.. |PyPI| image:: https://badge.fury.io/py/fitr.svg
.. _PyPI: https://badge.fury.io/py/fitr

.. |Codecov| image:: https://codecov.io/gh/ComputationalPsychiatry/fitr/branch/master/graphs/badge.svg
.. _Codecov: https://codecov.io/gh/ComputationalPsychiatry/fitr/branch/master

.. |Health| image:: https://landscape.io/github/ComputationalPsychiatry/fitr/master/landscape.svg?style=flat
.. _Health: https://landscape.io/github/ComputationalPsychiatry/fitr/master

.. |PyV| image:: https://img.shields.io/badge/python-3.5%2B-blue.svg
.. _PyV: https://badge.fury.io/py/fitr

.. |DOI| image:: https://zenodo.org/badge/82499710.svg
.. _DOI: https://zenodo.org/badge/latestdoi/82499710

fitr
====

Python implementation of package to fit reinforcement learning models to
behavioural data

Installation
------------

The current PyPI release of Fitr can be installed as follows::

    pip install fitr

If you want the latest version on the GitHub master branch, install as follows::

    pip install git+https://github.com/ComputationalPsychiatry/fitr.git

Currently, we build and test on Linux and OSX. As such, we cannot guarantee performance on Windows.

Documentation
-------------

`Fitr documentation <https://computationalpsychiatry.github.io/fitr/>`_ is hosted on GitHub Pages.

Tutorials
---------

Tutorials (Jupyter Notebooks) can be found in the examples folder. They include

1. `Introductory tutorial (EM and Bayesian Model Selection) <https://github.com/ComputationalPsychiatry/fitr/blob/master/examples/intro-tutorial.ipynb>`_
2. `Fitting a Model with MCMC <https://github.com/ComputationalPsychiatry/fitr/blob/master/examples/Fitting%20a%20Model%20with%20MCMC.ipynb>`_
3. `Use MCMC with your own Stan Code <https://github.com/ComputationalPsychiatry/fitr/blob/master/examples/Use%20MCMC%20with%20your%20own%20Stan%20Code.ipynb>`_
4. `Using Multiple Model-Fitting Routines for Same Model <https://github.com/ComputationalPsychiatry/fitr/blob/master/examples/Using%20Multiple%20Methods%20to%20fit%20Models.ipynb>`_

How to Cite
-----------

If you use Fitr in your work, we would very much appreciate the citation, which can be done as follows:

- Abraham Nunes, Alexander Rudiuk, & Thomas Trappenberg. (2017). Fitr: A Toolbox for Computational Psychiatry Research. Zenodo. http://doi.org/10.5281/zenodo.439989
