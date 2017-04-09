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
Objects representing each parameter object
"""
import numpy as np
import scipy.stats

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
        """

        return self.dist.rvs(size=size)

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

    """
    def __init__(self, name='Learning Rate', rng='unit', shape_alpha=1.1, shape_beta=1.1):
        """
        Instantiates the Parameter

        Parameters
        ----------
        name : str
            Name of the parameter. To be used for plots and so forth.
        rng : {'unit', 'pos', 'neg', 'unc'}
            The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
        shape_alpha : float over domain [0, +Inf]
            The alpha parameter of the beta distribution
        shape_beta : float over domain [0, +Inf]
            The beta parameter of the beta distribution
        """
        self.name = name
        self.rng  = rng
        self.dist = scipy.stats.beta(shape_alpha, shape_beta)

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

    """
    def __init__(self, name='Reward Sensitivity', rng='unit', shape_alpha=1.1, shape_beta=1.1):
        """
        Instantiates the Parameter

        Parameters
        ----------
        name : str
            Name of the parameter. To be used for plots and so forth.
        rng : {'unit', 'pos', 'neg', 'unc'}
            The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
        shape_alpha : float over domain [0, +Inf]
            The alpha parameter of the beta distribution
        shape_beta : float over domain [0, +Inf]
            The beta parameter of the beta distribution
        """
        self.name = name
        self.rng  = rng
        self.dist = scipy.stats.beta(shape_alpha, shape_beta)

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

    """
    def __init__(self, name='Eligibility Trace', rng='unit', shape_alpha=1.1, shape_beta=1.1):
        """
        Instantiates the Parameter

        Parameters
        ----------
        name : str
            Name of the parameter. To be used for plots and so forth.
        rng : {'unit', 'pos', 'neg', 'unc'}
            The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
        shape_alpha : float over domain [0, +Inf]
            The alpha parameter of the beta distribution
        shape_beta : float over domain [0, +Inf]
            The beta parameter of the beta distribution
        """
        self.name = name
        self.rng  = rng
        self.dist = scipy.stats.beta(shape_alpha, shape_beta)

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

    """
    def __init__(self, name='Model-Based Control Weight', rng='unit', shape_alpha=1.1, shape_beta=1.1):
        """
        Instantiates the Parameter

        Parameters
        ----------
        name : str
            Name of the parameter. To be used for plots and so forth.
        rng : {'unit', 'pos', 'neg', 'unc'}
            The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
        shape_alpha : float over domain [0, +Inf]
            The alpha parameter of the beta distribution
        shape_beta : float over domain [0, +Inf]
            The beta parameter of the beta distribution
        """

        self.name = name
        self.rng  = rng
        self.dist = scipy.stats.beta(shape_alpha, shape_beta)

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

    """
    def __init__(self, name='Choice Randomness', rng='pos', shape=5., scale=1.):
        """
        Instantiates the Parameter

        Parameters
        ----------
        name : str
            Name of the parameter. To be used for plots and so forth.
        rng : {'unit', 'pos', 'neg', 'unc'}
            The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
        shape : float over domain [0, +Inf]
            The shape parameter of the gamma distribution
        scale : float over domain [0, +Inf]
            The scale parameter of the gamma distribution
        """
        self.name = name
        self.rng  = rng
        self.dist = scipy.stats.gamma(shape, scale)

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


# ==============================================================================
#
#   SYNTHETIC GROUP
#       Generates a synthetic group of subjects
#
# ==============================================================================

def generate_group(params, nsubjects):
    """
    Creates an array of parameter values for subjects to be simulated in a task

    Parameters
    ----------
    params : list
        List of Param objects specifying the parameters to be generated. The first element of the list will be the first column in the resulting array
    nsubjects : int
        Number of subjects to simulate

    Returns
    -------
    ndarray(shape=(nsubjects X nparams))
        Array of parameter values for the group of synthetic subjects
    """

    nparams = len(params)

    group_params = np.zeros([nsubjects, nparams])

    for k in range(nparams):
        group_params[:, k] = params[k].sample(size=nsubjects)

    return group_params
