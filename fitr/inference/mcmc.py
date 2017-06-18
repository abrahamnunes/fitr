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
import pandas as pd
import pystan

from .modelfitresult import ModelFitResult

class MCMC(object):
    """
    Uses Markov-Chain Monte-Carlo (via PyStan) to estimate models

    Attributes
    ----------
    name : str
        Name of the model being fit
    generative_model : GenerativeModel object

    Methods
    -------
    fit(self, data, chains=4, n_iterations=2000, warmup=None, thin=1, seed=None, init='random', sample_file=None, algorithm='NUTS', control=None, n_jobs=-1, compile_verbose=False, sampling_verbose=False)
        Runs the MCMC Inference procedure with Stan

    """
    def __init__(self, generative_model=None, name='FitrMCMCModel'):
        self.name = name
        self.generative_model = generative_model

    def fit(self, data, chains=4, n_iterations=2000, warmup=None, thin=1, seed=None, init='random', sample_file=None, algorithm='NUTS', control=None, n_jobs=-1, compile_verbose=False, sampling_verbose=False):
        """
        Runs the MCMC Inference procedure with Stan

        Parameters
        ----------
        data : dict
            Subject level data
        chains : int > 0
            Number of chains in sampler
        n_iter : int
            How many iterations each chain should run (includes warmup)
        warmup : int > 0, iter//2 by default
            Number of warmup iterations.
        thin : int > 0
            Period for saving samples
        seed : int or np.random.RandomState, optional
            Positive integer to initialize random number generation
        sample_file : str
            File name specifying where samples for all parameters and other saved quantities will be written. If None, no samples will be written
        algorithm : {'NUTS', 'HMC', 'Fixed_param'}, optional
            Which of Stan's algorithms to implement
        control : dict, optional
            Dictionary of parameters to control sampler's behaviour (see PyStan documentation for details)
        n_jobs : int, optional
            Sample in parallel. If -1, all CPU cores are used. If 1, no parallel computing is used
        compile_verbose : bool
            Whether to print output from model compilation
        sampling_verbose : bool
            Whether to print intermediate output from model sampling

        Returns
        -------
        ModelFitResult
            Instance containing model fitting results

        References
        ----------
        .. [1] PyStan API documentation (https://pystan.readthedocs.io)
        """

        print('=============================================\n' +
              '     MODEL: ' + self.name + '\n' +
              '     METHOD: Markov Chain Monte-Carlo\n' +
              '     ITERATIONS: ' + str(n_iterations) + '\n' +
              '     OPTIMIZATION ALGORITHM: ' + algorithm + '\n' +
              '=============================================\n')

        # Instantiate a ModelFitResult object
        param_codenames = self.generative_model.paramnames['code']
        results = ModelFitResult(name=self.name,
                                 method='MCMC',
                                 nsubjects=data['N'],
                                 nparams=len(param_codenames))

        results.paramnames = self.generative_model.paramnames['long']

        model_code = self.generative_model.model

        # Compile generative model with Stan
        sm = pystan.StanModel(model_code=model_code,
                              verbose=compile_verbose)

        # Sample from generative model
        stanfit = sm.sampling(data=data,
                              chains=chains,
                              iter=n_iterations,
                              warmup=warmup,
                              thin=thin,
                              seed=seed,
                              init=init,
                              sample_file=sample_file,
                              verbose=sampling_verbose,
                              algorithm=algorithm,
                              control=control,
                              n_jobs=n_jobs)

        # Create summary dataframe
        summary_data = stanfit.summary()['summary']
        summary_colnames = stanfit.summary()['summary_colnames']
        summary_rownames = stanfit.summary()['summary_rownames']
        stan_summary = pd.DataFrame(data=summary_data,
                                    columns=summary_colnames,
                                    index=summary_rownames)

        # Extract parameter estimates for easy manipulation with ModelFitResult
        param_codes = self.generative_model.paramnames['code']
        param_est = stanfit.extract(pars=param_codes)

        # Get expected parameter estimates (subject-level) into params array
        param_idx = 0
        for k in self.generative_model.paramnames['code']:
            results.params[:, param_idx] = np.mean(param_est[k], axis=0)
            param_idx += 1

        # Populate results.stanfit dict with Stan related objects
        results.stanfit = {
            'stanfit' : stanfit,
            'summary' : stan_summary
        }

        return results
