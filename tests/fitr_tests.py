# -*- coding: utf-8 -*-

import fitr
from fitr import tasks
from fitr.models import twostep as task
from fitr import generative_models as gm
from fitr import loglik_functions as ll
import numpy as np
import scipy

def test_old_ll():
	ntrials=10
	nsubjects=5
	lr = fitr.rlparams.LearningRate()
	cr = fitr.rlparams.ChoiceRandomness()
	params = [lr, cr]

	group = np.zeros([nsubjects, 2])
	group[:, 0] = lr.sample(size=nsubjects)
	group[:, 1] = cr.sample(size=nsubjects)

	res = tasks.bandit(narms=2).simulate(params=group, ntrials=ntrials)

	L = ll.bandit_ll().lrp_lrn_cr(params=np.array([0.5, 0.5, 4]),
						     	  states=res.data[0]['S'],
							 	  actions=res.data[0]['A'],
							 	  rewards=res.data[0]['R'])
	assert(np.isfinite(L))

	L = ll.bandit_ll().dummy(params=np.array([4]),
					     	 states=res.data[0]['S'],
							 actions=res.data[0]['A'],
							 rewards=res.data[0]['R'])
	assert(np.isfinite(L))


def test_em():
	ntrials = 10
	nsubjects = 5

	res = task.lr_cr_mf().simulate(ntrials=ntrials, nsubjects=nsubjects)

	lr = fitr.rlparams.LearningRate()
	cr = fitr.rlparams.ChoiceRandomness()
	likfun = task.lr_cr_mf().loglikelihood

	model = fitr.fitr.EM(loglik_func=likfun,
						 params=[lr, cr],
						 name='EMModel')

	assert(model.name == 'EMModel')
	assert(len(model.params) == 2)
	assert(model.loglik_func == likfun)

	mfit = model.fit(data=res.data,
					 early_stopping=False)
	mfit2 = model.fit(data=res.data,
					  opt_algorithm='BFGS',
					  init_grid=True,
					  grid_reinit=True,
					  n_grid_points=5,
					  n_reinit=1,
					  dofull=False,
					  early_stopping=True,
					  verbose=False)

	mfit.plot_ae(actual=res.params)
	mfit.plot_fit_ts(s)
	mfit.param_hist()

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
	ntrials = 10
	nsubjects = 5

	lr = fitr.rlparams.LearningRate()
	cr = fitr.rlparams.ChoiceRandomness()

	res = task.lr_cr_mf().simulate(ntrials=ntrials, nsubjects=nsubjects)

	likfun = task.lr_cr_mf().loglikelihood

	model = fitr.fitr.EmpiricalPriors(loglik_func=likfun,
						 			  params=[lr, cr],
						 		  	  name='EPModel')

	assert(model.name == 'EPModel')
	assert(len(model.params) == 2)
	assert(model.loglik_func == likfun)

	mfit = model.fit(data=res.data)
	mfit2 = model.fit(data=res.data,
					  opt_algorithm='BFGS',
					  verbose=False)

	assert(mfit.name == 'EPModel')
	assert(mfit.method == 'Empirical Priors')
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
	nsubjects = 5
	ntrials = 10

	lr = fitr.rlparams.LearningRate()
	cr = fitr.rlparams.ChoiceRandomness()
	params = [lr, cr]

	group = np.zeros([nsubjects, 2])
	group[:, 0] = lr.sample(size=nsubjects)
	group[:, 1] = cr.sample(size=nsubjects)

	taskresults = tasks.bandit(narms=2).simulate(params=group, ntrials=ntrials)
	banditgm = gm.bandit(model='lr_cr')

	model = fitr.MCMC(generative_model=banditgm)

	assert(model.name == 'FitrMCMCModel')
	assert(model.generative_model == banditgm)

	lrcr = model.fit(data=taskresults.data_mcmc, n_iterations=10)
	lrcr.trace_plot()

	assert(lrcr.name == 'FitrMCMCModel')
	assert(lrcr.method == 'MCMC')
	assert(lrcr.nsubjects == 5)
	assert(lrcr.nparams == 2)
	assert(np.shape(lrcr.params) == (5, 2))
	assert(len(lrcr.paramnames) == 2)
	assert(type(lrcr.stanfit) == dict)

def test_fitrmodels():
	nsubjects = 5
	ntrials = 10

	lr = fitr.rlparams.LearningRate()
	cr = fitr.rlparams.ChoiceRandomness()
	params = [lr, cr]

	group = np.zeros([nsubjects, 2])
	group[:, 0] = lr.sample(size=nsubjects)
	group[:, 1] = cr.sample(size=nsubjects)

	taskresults = tasks.bandit(narms=2).simulate(params=group, ntrials=ntrials)

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
