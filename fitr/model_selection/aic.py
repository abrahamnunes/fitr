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

import numpy as np

from scipy.special import digamma as psi
from scipy.special import gammaln

from .modelselectionresult import ModelSelectionResult

class AIC(object):
    """
    Model comparison with Aikake Information Criterion

    Attributes
    ----------
    modelfits : list
        List of fitrfit objects from completed model fitting

    Methods
    -------
    run(self)
        Runs model comparison by Aikake Information Criterion

    """
    def __init__(self, model_fits):
        self.modelfits = model_fits

    def run(self):
        """
        Runs model comparison by Aikake Information Criterion
        """
        results = ModelSelectionResult(method='AIC')

        for i in range(len(self.modelfits)):
            results.AIC.append(np.sum(self.modelfits[i].AIC))

        for i in range(len(self.modelfits)):
            results.modelnames.append(self.modelfits[i].name)

        return results
