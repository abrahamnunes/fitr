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

from .em import EM
from .empiricalpriors import EmpiricalPriors
from .mcmc import MCMC
from .mle import MLE

class FitModel(object):
    """
    An object representing a model to be fit to behavioural data. This should be viewed as a high level wrapper for multiple potential model fitting algorithms which themselves can be run by using their respective classes.

    Attributes
    ----------
    name : str
        Name of the model. We suggest identifying model based on free parameters.
    loglik_func : function
        The log-likelihood function to be used to fit the data
    params : list
        List of reinforcement learning parameter objects from the rlparams module.
    generative_model : GenerativeModel object
        Object representing a generative model

    Methods
    -------
    fit(data, method='EM', c_limit=0.01)
        Runs the specified model fitting algorithm with the given data.
    """
    def __init__(self, name='Anon Model', loglik_func=None, params=None, generative_model=None):
        self.name = name
        self.loglik_func = loglik_func
        self.generative_model = generative_model
        self.params = params

    def fit(self, data, method='EM', c_limit=0.01, verbose=True):
        """
        Runs model fitting

        Parameters
        ----------
        data : dict
            Behavioural data.
        method : {'EM', 'MLE', 'EmpiricalPriors', 'MCMC'}
            The inference algorithm to use. Note that the data formats for 'MCMC' compared to the other methods is distinct, and should correspond appropriately to the method being employed
        c_limit : float
            Limit at which convergence of log-posterior probability is determined (only for methods 'EM' and 'EmpiricalPriors')
        verbose : bool
            Controls amount of printed output during model fitting

        Returns
        -------
        fitrfit : object
            Representation of the model fitting results
        """

        if method=='EM':
            m = EM(loglik_func=self.loglik_func,
                   params=self.params,
                   name=self.name)
            results = m.fit(data=data,
                            c_limit=c_limit,
                            verbose=verbose)
        elif method=='MLE':
            m = MLE(loglik_func=self.loglik_func,
                    params=self.params,
                    name=self.name)
            results = m.fit(data=data,
                            n_iterations=1,
                            c_limit=c_limit,
                            verbose=verbose)
        elif method=='EmpiricalPriors':
            m = EmpiricalPriors(loglik_func=self.loglik_func,
                                params=self.params,
                                name=self.name)
            results = m.fit(data=data,
                            c_limit=c_limit,
                            verbose=verbose)
        elif method=='MCMC':
            m = MCMC(generative_model=self.generative_model, name=self.name)
            results = m.fit(data=data)

        return results
