Contributing to Fitr
====================

Your contributions to Fitr are welcome and encouraged. Fitr is being developed on GitHub in order to facilitate improvements by the community. However, to ensure that Fitr develops as a robust piece of software, we have several guidelines for contributions. These have been chosen to mirror those of the SciPy/NumPy project.

Contributions to Fitr should have

1. Unit tests
    * It is important that Fitr functions well "out of the box," and this requires that code implemented in Fitr is appropriately tested.
    * Fitr uses Codecov.io to assess code coverage. In general, try to ensure that your new code is covered by unit tests.
    * Unit tests are written in the ``/tests`` folder, where you can find examples of how unit tests are currently written.
2. Documentation
    * New code is not of great use unless the community knows what it is for and how to use it. As such, we ask that any new functions or modifications to existing functions carry the appropriate documentation.
    * If your contribution is substantial, it may be of use to write a tutorial, which are done with Jupyter Notebooks here.
    * Documentation of modules, classes, and functions can be done in Docstrings, then compiled with Sphinx and autodoc.
    * Documentation of Fitr code follows the `SciPy/NumPy format <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
3. Appropriate code style
    * Fitr follows the `PEP8 <http://www.python.org/dev/peps/pep-0008/>`_ standard, and so we recommend that you run a linter to ensure that contributions adhere to this format.

Types of Contributions
----------------------

At this early stage, we are open to any new contributions, big or small.

Many of the contribution requirements listed above were not adhered to at Fitr's inception, so even if you would like to help by correcting some of our past mistakes, this would be an important step toward Fitr's goals!

How to Contribute
-----------------

1. Fork the GitHub repository
2. Create a new branch
3. Submit a pull request

Fitr's master branch is protected and requires Unit tests to pass, as well as 2 reviews before merging is allowed.

Requesting Features and Reporting Problems
------------------------------------------

Open an issue on the Fitr GitHub page, and we'll get on it as soon as possible!
