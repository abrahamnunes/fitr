# -*- coding: utf-8 -*-

import fitr
from fitr import tasks
from fitr import generative_models as gm
from fitr import loglik_functions as ll
import numpy as np
import scipy

def test_em():
	lr = fitr.rlparams.LearningRate()
	cr = fitr.rlparams.ChoiceRandomness()
	params = [lr, cr]
	group = fitr.rlparams.generate_group(params=params, nsubjects=5)
	bandit_task = tasks.bandit()
	res = bandit_task.simulate(ntrials=10, params=group)

	likfun = ll.bandit_ll().lr_cr

	model = fitr.fitr.EM(loglik_func=likfun,
						 params=params,
						 name='EMModel')

	assert(model.name == 'EMModel')
	assert(len(model.params) == 2)
	assert(model.loglik_func == likfun)

	mfit = model.fit(data=res.data)

	assert(mfit.name == 'EMModel')
	assert(mfit.method == 'Expectation-Maximization')
	assert(mfit.nsubjects == 5)
	assert(mfit.nparams == 2)
	assert(np.shape(mfit.params) == (5, 2))
	assert(len(mfit.paramnames) == 2)
	assert(np.shape(mfit.hess) == (2, 2, 5))
	assert(np.shape(mfit.hess_inv) == (2, 2, 5))
	assert(np.shape(mfit.errs) == (5, 2))
	assert(np.size(mfit.nlogpost) == 5)
	assert(np.size(mfit.nloglik) == 5)
	assert(np.size(mfit.LME) == 5)
	assert(np.size(mfit.BIC) == 5)
	assert(np.size(mfit.AIC) == 5)
	assert(type(mfit.ts_LME) == list)
	assert(type(mfit.ts_nLL) == list)
	assert(type(mfit.ts_BIC) == list)
	assert(type(mfit.ts_AIC) == list)

def test_empirical_priors():
	lr = fitr.rlparams.LearningRate()
	cr = fitr.rlparams.ChoiceRandomness()
	params = [lr, cr]
	group = fitr.rlparams.generate_group(params=params, nsubjects=5)
	bandit_task = tasks.bandit()
	res = bandit_task.simulate(ntrials=10, params=group)

	likfun = ll.bandit_ll().lr_cr

	model = fitr.fitr.EmpiricalPriors(loglik_func=likfun,
						 			  params=params,
						 		  	  name='EPModel')

	assert(model.name == 'EPModel')
	assert(len(model.params) == 2)
	assert(model.loglik_func == likfun)

	mfit = model.fit(data=res.data)

	assert(mfit.name == 'EMModel')
	assert(mfit.method == 'Expectation-Maximization')
	assert(mfit.nsubjects == 5)
	assert(mfit.nparams == 2)
	assert(np.shape(mfit.params) == (5, 2))
	assert(len(mfit.paramnames) == 2)
	assert(np.shape(mfit.hess) == (2, 2, 5))
	assert(np.shape(mfit.hess_inv) == (2, 2, 5))
	assert(np.shape(mfit.errs) == (5, 2))
	assert(np.size(mfit.nlogpost) == 5)
	assert(np.size(mfit.nloglik) == 5)
	assert(np.size(mfit.LME) == 5)
	assert(np.size(mfit.BIC) == 5)
	assert(np.size(mfit.AIC) == 5)
	assert(type(mfit.ts_LME) == list)
	assert(type(mfit.ts_nLL) == list)
	assert(type(mfit.ts_BIC) == list)
	assert(type(mfit.ts_AIC) == list)

def test_mcmc():
	params = [fitr.rlparams.LearningRate(),
	 		  fitr.rlparams.ChoiceRandomness()]
	group = generate_group(params=params, nsubjects=5)
	taskresults = tasks.bandit(narms=2).simulate(params=group, ntrials=10)
	banditgm = gm.bandit(model='lr_cr')

	model = fitr.MCMC(generative_model=banditgm)

	assert(model.name == 'FitrMCMCModel')
	assert(model.generative_model == banditgm)

	lrcr = model.fit(data=taskresults.data_mcmc, n_iterations=10)

	assert(lrcr.name == 'FitrMCMCModel')
	assert(lrcr.method == 'MCMC')
	assert(lrcr.nsubjects == 5)
	assert(lrcr.nparams == 2)
	assert(np.shape(lrcr.params) == (5, 2))
	assert(len(lrcr.paramnames) == 2)
	assert(type(lrcr.stanfit) == dict)

def test_fitrmodels():
	params = [fitr.rlparams.LearningRate(),
	 		  fitr.rlparams.ChoiceRandomness()]
	group = generate_group(params=params, nsubjects=5)
	taskresults = tasks.bandit(narms=2).simulate(params=group, ntrials=10)

	banditll = ll.bandit_ll().lr_cr
	banditgm = gm.bandit(model='lr_cr')

	model = fitr.fitrmodel(name='My 2-Armed Bandit Model',
                       	   loglik_func=banditll,
                       	   params=params,
                       	   generative_model=banditgm)

	emfit = model.fit(data=taskresults.data,
					  method='EM',
					  verbose=False)

	epfit = model.fit(data=taskresults.data,
					  method='EmpiricalPriors',
					  verbose=False)

	mcfit = model.fit(data=taskresults.data_mcmc,
					  method='MCMC',
					  verbose=False)
