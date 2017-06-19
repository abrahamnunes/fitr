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
Module containing clustering algorithms

Module Documentation
--------------------
"""

import numpy as np
import pandas as pd

from sklearn.cluster import AffinityPropagation as AP
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score

class Cluster(object):
    """
    Base clustering class

    Attributes
    ----------
    algorithm : ``sklearn.cluster`` class
        The algorithm to be used for clustering
    clusters : pandas.DataFrame
        Contains true and estimated labels
    performance : pandas.DataFrame
        Performance statistics for clustering algorithms

    Methods
    -------
    plot_distance()
        Returns an image plot of the distance matrix
    performance()
        Returns performance metrics for clustering algorithm

    """
    def __init__(self):
        self.algorithm = None

        self.clusters = None
        self.results = None

class AffinityPropagation(Cluster):
    """
    Wrapper for ``sklearn.cluster.AffinityPropagation``
    """
    def __init__(self, damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='precomputed', verbose=False):
        self.algorithm = AP(damping=damping,
                            max_iter=max_iter,
                            convergence_iter=convergence_iter,
                            copy=copy,
                            preference=preference,
                            affinity=affinity,
                            verbose=verbose)

    def fit(self, data):
        """
        Runs the affinity propagation clustering algorithm

        Parameters
        ----------
        data : ndarray
            Data to be clustered
        """

        self.algorithm.fit(data)

        # Extract the cluster labels
        labels = self.algorithm.labels_

        self.clusters = pd.DataFrame(data = {
            'labels'   : labels
            }, index=None)

        # Extract the number of clusters
        nclusters = len(self.algorithm.cluster_centers_indices_)

        self.results = pd.DataFrame(data={
            'nclusters' : [nclusters]
        })

    def performance(self, group_labels=None):
        """
        Computes performance metrics for clustering algorithm

        Parameters
        ----------
        group_labels : (optional) ndarray(shape=nsubjects)
            Labels for subject groups
        """
        n_samples = len(self.algorithm.labels_)

        if group_labels is None:
            truelab = np.zeros(n_samples)
            unique_labels = np.unique(group_labels)
            self.clusters["true_int"] = truelab
        else:
            truelab = np.zeros(n_samples)
            unique_labels = np.unique(group_labels)

            for i, label_i in enumerate(unique_labels):
                truelab[group_labels == label_i] = i

            self.clusters["true"] = group_labels
            self.clusters["true_int"] = truelab

        lab = self.algorithm.labels_
        self.results["homogeneity"] = homogeneity_score(truelab, lab)
        self.results["completeness"] = completeness_score(truelab, lab)
        self.results["v_measure"] = v_measure_score(truelab, lab)
        self.results["adj_rand"] = adjusted_rand_score(truelab, lab)
        self.results["adj_MI"] = adjusted_mutual_info_score(truelab, lab)
