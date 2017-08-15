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
Statistics for Comparing True and Estimated Parameters

Module Documentation
--------------------
"""
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import ttest_ind


def paramcorr(X, Y):
    """
    Pearson correlation coefficient for actual and estimated parameters

    Parameters
    ----------
    X : ndarray(shape=(n_subjects, nparams))
        First array of parameters to be correlated
    Y : ndarray(shape=(n_subjects, nparams))
        Second array of parameters to be correlated

    Returns
    -------
    corrs : ndarray(shape=(n_params, 2))
        Array of correlation coefficients and p-values

    Notes
    -----
    Arrays ``X`` and ``Y`` must be the same size
    """
    nparams = np.shape(X)[1]
    corrs = np.zeros([nparams, 2])

    for j in range(nparams):
        corrs[j, :] = pearsonr(X[:,j], Y[:,j])

    return corrs

def param_ttest(X, Y):
    """
    Two-sample t-test for difference between parameters (actual and estimated)

    Parameters
    ----------
    X : ndarray(shape=(n_subjects, nparams))
    Y : ndarray(shape=(n_subjects, nparams))

    Returns
    -------
    res : ndarray(shape=(n_params, 2))
        (t-statistic, p-value)

    Notes
    -----
    Arrays ``X`` and ``Y`` must be the same size
    """
    nparams = np.shape(X)[1]
    res = np.zeros([nparams, 2])

    for j in range(nparams):
        res[j, :] = ttest_ind(X[:,j], Y[:,j])

    return res
