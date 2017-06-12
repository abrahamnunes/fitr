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
Distance measures

Module Documentation
--------------------
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

def parameter_distance(params, dist_metric='canberra', scale='minmax', return_scaled=False):
    """
    Computes distances between subjects' respective parameter estimates

    Parameters
    ----------
    params : ndarray(shape=(nsubjects, nsubjects))
        Array of parameter estimates
    dist_metric : str (default='canberra')
        Distance metric to be used. Can take any value acceptable by ``sklearn.metrics.pairwise_distances``.
    scale : {'minmax', 'standard', 'none'}
        How to scale the parameters for distance computation
    return_scaled : bool
        Whether to return scaled parameters
    """

    if scale != 'none':
        if scale == 'minmax':
            scaler = MinMaxScaler()
        if scale == 'standard':
            scaler = StandardScaler()

        nparams = np.shape(params)[1]
        for j in range(nparams):
            scaledparam = scaler.fit_transform(params[:, j].reshape(-1, 1))
            params[:, j] = scaledparam.flatten()

    if return_scaled is True:
        D = (pairwise_distances(params, metric=dist_metric), params)
    else:
        D = pairwise_distances(params, metric=dist_metric)

    return D

def likelihood_distance(loglik_func, data, params, diff_metric='sq', dist_metric='cosine', verbose=False):
    """
    Estimates the likelihood of the data from the i'th subject using the parameter estimates of the j'th subject, for all i and j, then computes the distance between subjects' likelihood difference vectors

    Parameters
    ----------
    loglik_func : function
        The log-likelihood function to be used
    data : dict
        Data formatted for input into the log-likelihood function
    params : ndarray(shape=(nsubjects, nparams))
        Array of parameter estimates
    diff_metric : {'sq', 'diff', 'abs'}
        Which type of difference measure to compute, 'diff' is simple subtractive difference, whereas 'sq' and 'abs' are the squared and absolute differences, respectively
    dist_metric : str (default='cosine')
        The pairwise distance metric to use. Any option that can be passed into ``sklearn.metrics.pairwise_distances`` can work.
    verbose : bool
        Whether to print out progress

    Returns
    -------
    ndarray(shape=(nsubjects, nsubjects))
    """
    nsubjects = np.shape(params)[0]
    D = np.zeros([nsubjects, nsubjects])
    for i in range(nsubjects):
        S = data[i]['S']
        A = data[i]['A']
        R = data[i]['R']

        if verbose is True:
            print('Likelihood Differences: Subject ' + str(i))

        # Compute loglikelihood for subject i with own data
        LL0 = loglik_func(params=params[i, :],
                          states=S,
                          actions=A,
                          rewards=R)

        for j in range(nsubjects):
            if i !=j:
                LL1 = loglik_func(params=params[j, :],
                                  states=S,
                                  actions=A,
                                  rewards=R)

                if diff_metric == 'diff':
                    D[i, j] = LL1 - LL0
                elif diff_metric == 'sq':
                    D[i, j] = (LL1 - LL0)**2
                elif diff_metric == 'abs':
                    D[i, j] = np.abs(LL1 - LL0)

    return pairwise_distances(D, metric=dist_metric)
