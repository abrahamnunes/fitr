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

def BIC(loglik, nparams, nsteps):
    """
    Calculates Bayesian information criterion

    Parameters
    ----------
    loglik : float or ndarray(dtype=float)
        Log-likelihood
    nparams : int
        Number of parameters in the model
    nsteps : int
        Number of time steps in the task

    Returns
    -------
    float or ndarray(dtype=float)

    """
    return nparams*np.log(nsteps) - 2*loglik

def AIC(nparams, loglik):
    """
    Calculates Aikake information criterion

    Parameters
    ----------
    nparams : int
        Number of parameters in the model
    loglik : float or ndarray(dtype=float)
        Log-likelihood

    Returns
    -------
    float or ndarray(dtype=float)

    """
    return 2*nparams - 2*loglik

def LME(logpost, nparams, hessian):
    """
    Calculates log-model-evidence (LME)

    Parameters
    ----------
    logpost : float or ndarray(dtype=float)
        Log-posterior probability
    nparams : int
        Number of parameters in the model
    hessian : ndarray(size=(nparams, nparams))
        Hessian computed from parameter optimization

    Returns
    -------
    float or ndarray(dtype=float)
    """
    return logpost + (nparams/2)*np.log(2*np.pi)-np.log(np.linalg.det(hessian))/2
