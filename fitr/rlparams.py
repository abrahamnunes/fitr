# -*- coding: utf-8 -*-
# Fitr. A package for fitting reinforcement learning models to behavioural data
# Copyright (C) 2017 Abraham Nunes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# CONTACT INFO:
#   Abraham Nunes
#    Email: nunes@dal.ca
#
# ============================================================================

"""
Module containing commonly used reinforcement learning parameter objects.

Module Documentation
--------------------
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# ==============================================================================
#
#   PARAMETER OBJECTS
#
# ==============================================================================


class Param(object):
    """
    A base parameter object that can be used to generate new parameters.

    Attributes
    ----------
    name : str
        Name of the parameter. To be used for plots and so forth.
    rng : {'unit', 'pos', 'neg', 'unc'}
        The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])

    Methods
    -------
    sample(size=1)
        Samples from the parameter's distribution

    """
    def __init__(self, name=None, rng=None):
        self.name  = name
        self.rng = rng
        self.dist  = None

    def sample(self, size=1):
        """
        Samples from the parameter's distribution

        Parameters
        ----------
        size : int
            Number of samples to draw

        Returns
        -------
        ndarray

        """

        return self.dist.rvs(size=size)

    def convert_meansd(self, mean, sd, dist):
        """
        Converts mean and standard deviation to other distribution parameters.

        Parameters
        ----------
        mean : float
            Mean value for distribution (must lie within region of support)
        sd : float
            Standard deviation for distribution
        dist : {'beta', 'gamma'}
            Target distribution

        Notes
        -----
        Currently, only the gamma and beta distributions are supported for this function.

        The Beta distribution has two shape parameters :math:`\lbrace \alpha, \beta \rbrace > 0`. Using the mean :math:`\mu` and the standard deviation :math:`\sigma`, the :math:`\alpha` parameter can be calculated as

        .. math::

            \alpha = (\frac{1-\mu}{\sigma^2} - \frac{1}{\mu})\mu^2,

        and the :math:`\beta` parameter as

        .. math::

            \beta = \alpha (\frac{1}{\mu} - 1).

        Note that for the Beta distribution to be defined this way, the following constraint must hold for the mean, :math:`0 < \mu < 1, and the following for the variance, :math:`0 < \sigma^2 \leq \mu - \mu^2`

        For the Gamma distribution, we have a shape parameter :math:`\kappa > 0` and a scale parameter :math:`\theta`. These can be calculated using the mean :math:`\mu` and standard deviation :math:`\sigma` as

        .. math::

            \theta = \frac{\sigma^2}{\mu},

        and

        .. math::

            \kappa = \frac{\mu^2}{\sigma^2}
        """

        if sd <= 0:
            raise ValueError('Standard deviation must be greater than 0.')
        else:
            if dist == 'beta':
                if mean <= 0 or mean >= 1:
                    raise ValueError('Mean for beta distribution must lie between 0 and 1.')
                elif sd**2 >= (mean - mean**2):
                    raise ValueError('Standard deviation must be greater than zero, but less than (mean - mean^2).')
                else:
                    shape_alpha =((1-mean)/(sd**2) - 1/mean)*mean**2
                    shape_beta = shape_alpha*(1/mean - 1)
                    self.dist = scipy.stats.beta(shape_alpha, shape_beta)
            elif dist == 'gamma':
                if mean <= 0:
                    raise ValueError('Mean for gamma distribution must be greater than 0.')
                else:
                    shape_kappa = mean**2/sd**2
                    shape_theta = sd**2/mean
                    self.dist = scipy.stats.gamma(a=shape_kappa, scale=shape_theta)
            else:
                raise ValueError('Only beta and gamma distributions are supported. Please see documentation for details.')

    def plot_pdf(self, xlim=None, figsize=None, show_figure=True, save_figure=False, filename='parameter-pdf.pdf'):
        """
        Plots the probability density function of this parameter

        Parameters
        ----------
        xlim : (optional) list of lower and upper bounds of x axis
        figsize : (optional) list defining plot dimensions
        show_figure : bool
            Whether to show the figure on function call
        save_figure : bool
            Whether to save the figure at function call
        filename : str
            The name of the file at which to save the figure

        """
        if figsize is None:
            figsize = (5, 5)

        if xlim is None:
            if self.rng == 'unit':
                lb = 0
                ub = 1
            if self.rng == 'pos':
                lb = 0
                ub = self.dist.mean() + 6*self.dist.std()
            if self.rng == 'neg':
                lb = self.dist.mean() - 6*self.dist.std()
                ub = 0
            if self.rng == 'unc':
                lb = self.dist.mean() - 6*self.dist.std()
                ub = self.dist.mean() + 6*self.dist.std()
        else:
            lb = xlim[0]
            ub = xlim[1]
            if lb > ub:
                raise ValueError('Lower bound must be lower than upper bound on plotting range.')
            if self.rng == 'unit':
                if lb < 0 or ub > 1 or ub < 0:
                    raise ValueError('Plotting range for parameters on the unit interval must have bounds between 0 and 1')
            elif self.rng == 'pos':
                if lb < 0 or ub < 0:
                    raise ValueError('Bounds of plotting range for a parameter on positive real line must both be >=0.')
            elif self.rng == 'neg':
                if lb > 0 or ub > 0:
                    raise ValueError('Bounds of plotting range for a parameter on negative real line must both be <=0.')

        X = np.linspace(lb, ub, 200)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(X, self.dist.pdf(X), c='k')
        ax.set_xlabel(self.name + ' Value')
        ax.set_title(self.name + ' PDF')

        if save_figure is True:
            plt.savefig(filename, bbox_inches='tight')

        if show_figure is True:
            plt.show()


class LearningRate(Param):
    """
    A learning rate object.

    Attributes
    ----------
    name : str
        Name of the parameter. To be used for plots and so forth.
    rng : {'unit', 'pos', 'neg', 'unc'}
        The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
    dist : scipy.stats.beta distribution

    Methods
    -------
    sample(size=1)
        Samples from the parameter's distribution

    """
    def __init__(self, name='Learning Rate', rng='unit', mean=0.5, sd=0.27):
        """
        Instantiates the Parameter

        Parameters
        ----------
        name : str
            Name of the parameter. To be used for plots and so forth.
        rng : {'unit', 'pos', 'neg', 'unc'}
            The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
        mean : float on interval (0, 1)
            Mean of the distribution
        sd : float on interval (0, +Inf)
            Standard deviation of the distribution
        """
        self.name = name
        self.rng  = rng

        self.convert_meansd(mean=mean, sd=sd, dist='beta')

class RewardSensitivity(Param):
    """
    A reward sensitivity object.

    Attributes
    ----------
    name : str
        Name of the parameter. To be used for plots and so forth.
    rng : {'unit', 'pos', 'neg', 'unc'}
        The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
    dist : scipy.stats.beta distribution

    Methods
    -------
    sample(size=1)
        Samples from the parameter's distribution

    """
    def __init__(self, name='Reward Sensitivity', rng='unit', mean=0.5, sd=0.27):
        """
        Instantiates the Parameter

        Parameters
        ----------
        name : str
            Name of the parameter. To be used for plots and so forth.
        rng : {'unit', 'pos', 'neg', 'unc'}
            The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
        mean : float on interval (0, 1)
            Mean of the distribution
        sd : float on interval (0, +Inf)
            Standard deviation of the distribution
        """
        self.name = name
        self.rng  = rng

        self.convert_meansd(mean=mean, sd=sd, dist='beta')

class EligibilityTrace(Param):
    """
    An eligibility trace parameter object.

    Attributes
    ----------
    name : str
        Name of the parameter. To be used for plots and so forth.
    rng : {'unit', 'pos', 'neg', 'unc'}
        The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
    dist : scipy.stats.beta distribution

    Methods
    -------
    sample(size=1)
        Samples from the parameter's distribution

    """
    def __init__(self, name='Eligibility Trace', rng='unit', mean=0.5, sd=0.27):
        """
        Instantiates the Parameter

        Parameters
        ----------
        name : str
            Name of the parameter. To be used for plots and so forth.
        rng : {'unit', 'pos', 'neg', 'unc'}
            The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
        mean : float on interval (0, 1)
            Mean of the distribution
        sd : float on interval (0, +Inf)
            Standard deviation of the distribution
        """
        self.name = name
        self.rng  = rng

        self.convert_meansd(mean=mean, sd=sd, dist='beta')

class MBMF_Balance(Param):
    """
    An object representing the parameter that balances model-based and model-free control.

    Attributes
    ----------
    name : str
        Name of the parameter. To be used for plots and so forth.
    rng : {'unit', 'pos', 'neg', 'unc'}
        The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
    dist : scipy.stats.beta distribution

    Methods
    -------
    sample(size=1)
        Samples from the parameter's distribution

    """
    def __init__(self, name='Model-Based Control Weight', rng='unit', mean=0.5, sd=0.27):
        """
        Instantiates the Parameter

        Parameters
        ----------
        name : str
            Name of the parameter. To be used for plots and so forth.
        rng : {'unit', 'pos', 'neg', 'unc'}
            The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
        mean : float on interval (0, 1)
            Mean of the distribution
        sd : float on interval (0, +Inf)
            Standard deviation of the distribution
        """
        self.name = name
        self.rng  = rng

        self.convert_meansd(mean=mean, sd=sd, dist='beta')

class ChoiceRandomness(Param):
    """
    An choice randomness parameter object

    Attributes
    ----------
    name : str
        Name of the parameter. To be used for plots and so forth.
    rng : {'unit', 'pos', 'neg', 'unc'}
        The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
    dist : scipy.stats.gamma distribution

    Methods
    -------
    sample(size=1)
        Samples from the parameter's distribution

    """
    def __init__(self, name='Choice Randomness', rng='pos', mean=4, sd=1):
        """
        Instantiates the Parameter

        Parameters
        ----------
        name : str
            Name of the parameter. To be used for plots and so forth.
        rng : {'unit', 'pos', 'neg', 'unc'}
            The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
        mean : float on interval (0, +Inf)
            Mean of the distribution
        sd : float on interval (0, +Inf)
            Standard deviation of the distribution
        """
        self.name = name
        self.rng  = rng

        self.convert_meansd(mean=mean, sd=sd, dist='gamma')

class Perseveration(Param):
    """
    An perseveration parameter object

    Attributes
    ----------
    name : str
        Name of the parameter. To be used for plots and so forth.
    rng : {'unit', 'pos', 'neg', 'unc'}
        The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
    dist : scipy.stats.norm distribution

    Methods
    -------
    sample(size=1)
        Samples from the parameter's distribution

    """
    def __init__(self, name='Perseveration', rng='unc', mean=0., sd=1.):
        """
        Instantiates the Parameter

        Parameters
        ----------
        name : str
            Name of the parameter. To be used for plots and so forth.
        rng : {'unit', 'pos', 'neg', 'unc'}
            The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
        mean : float
            The mean of the Gaussian distribution
        scale : float over domain [0, +Inf]
            The standard deviation of the Gaussian distribution
        """
        self.name = name
        self.rng  = rng
        self.dist = scipy.stats.norm(loc=mean, scale=sd)
