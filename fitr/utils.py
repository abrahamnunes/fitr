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
Module containing functions that are used across Fitr modules

References
----------
.. [Akam2015] Akam, T. et al. (2015) Simple Plans or Sophisticated Habits? State, Transition and Learning Interactions in the Two-Step Task. PLoS Comput. Biol. 11, 1–25

Module Documentation
--------------------
"""
import numpy as np

def softmax(x):
    """
    Computes numerically stable softmax

    Parameters
    ----------
    x : ndarray(shape=(nactions))

    Returns
    -------
    ndarray(shape=(nactions))
        Softmax probabilities for each action

    """
    xmax = np.max(x)
    return np.exp(x-xmax)/np.sum(np.exp(x-xmax))

def action(x):
    """
    Selects an action based on state-action values

    Parameters
    ----------
    x : ndarray
        Array of action values (scaled by inverse softmax temperature).

    Returns
    -------
    int
        The index corresponding to the selected action

    Notes
    -----
    This function computes the softmax probability for each action in the input array, and subsequently samples from a multinomial distribution parameterized by the results of the softmax computation. Finally, it returns the index where the value is equal to 1 (i.e. which action was selected).

    """
    p = np.exp(x) / np.sum(np.exp(x))
    return np.argmax(np.random.multinomial(1, pvals=p))

def logsumexp(x):
    """
    Numerically stable logsumexp.

    Parameters
    ----------
    x : ndarray(shape=(nactions))

    Returns
    -------
    float

    Notes
    -----
    The numerically stable log-sum-exp is computed as follows:

    .. math:: \max X + \log \sum_X e^{X - \max X}
    """
    xmax = np.max(x)
    y = xmax + np.log(np.sum(np.exp(x-xmax)))
    return y

def trans_UC(values_U, rng):
    """
    Transforms parameters from unconstrained to constrained space

    Parameters
    ----------
    values_U : ndarray
        Parameter values
    rng : {'unit', 'pos', 'half', 'all_unc'}
        The constrained range of the parameter

    Returns
    -------
    ndarray(shape=(nparams))

    Notes
    -----
    This code was taken from that published along with [Akam2015]_.
    """
    if rng[0] == 'all_unc':
        return values_U
    values_T = []
    for value, rng in zip(values_U, rng):
        if rng   == 'unit':  # Range: 0 - 1.
            if value < -16.:
                value = -16.
            values_T.append(1./(1. + np.exp(-value)))  # Don't allow values smaller than 1e-
        elif rng   == 'half':  # Range: 0 - 0.5
            if value < -16.:
                value = -16.
            values_T.append(0.5/(1. + np.exp(-value)))  # Don't allow values smaller than 1e-7
        elif rng == 'pos':  # Range: 0 - inf
            if value > 16.:
                value = 16.
            values_T.append(np.exp(value))  # Don't allow values bigger than ~ 1e7.
        elif rng == 'unc': # Range: - inf - inf.
            values_T.append(value)
    return np.array(values_T)
