# -*- coding: utf-8 -*-

import fitr
from fitr.inference import FitModel
from fitr.inference import EM
from fitr.inference import EmpiricalPriors
from fitr.inference import MCMC
from fitr.inference import MLE
from fitr.models import twostep as task
from fitr.models import driftbandit as db
from fitr import generative_models as gm
import numpy as np
import scipy

def test_em():
	ntrials = 100
	nsubjects = 5

	res = task.lr_cr_mf().simulate(ntrials=ntrials, nsubjects=nsubjects)

	lr = fitr.rlparams.LearningRate()
	cr = fitr.rlparams.ChoiceRandomness()
	likfun = task.lr_cr_mf().loglikelihood

	model = EM(loglik_func=likfun,
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

	mfit.plot_ae(actual=res.params, save_figure=True)
	mfit.plot_fit_ts(save_figure=True)
	mfit.param_hist(save_figure=True)

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
	ntrials = 100
	nsubjects = 5

	lr = fitr.rlparams.LearningRate()
	cr = fitr.rlparams.ChoiceRandomness()

	res = task.lr_cr_mf().simulate(ntrials=ntrials, nsubjects=nsubjects)

	likfun = task.lr_cr_mf().loglikelihood

	model = EmpiricalPriors(loglik_func=likfun,
						 	params=[lr, cr],
						 	name='EPModel')

	assert(model.name == 'EPModel')
	assert(len(model.params) == 2)
	assert(model.loglik_func == likfun)

	mfit = model.fit(data=res.data, verbose=False)
	mfit2 = model.fit(data=res.data,
					  opt_algorithm='BFGS',
					  verbose=True)

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

def test_mle():
	ntrials = 100
	nsubjects = 5

	lr = fitr.rlparams.LearningRate()
	cr = fitr.rlparams.ChoiceRandomness()

	res = task.lr_cr_mf().simulate(ntrials=ntrials, nsubjects=nsubjects)

	likfun = task.lr_cr_mf().loglikelihood

	model = MLE(loglik_func=likfun,
				params=[lr, cr],
				name='MLModel')

	assert(model.name == 'MLModel')
	assert(len(model.params) == 2)
	assert(model.loglik_func == likfun)

	mfit = model.fit(data=res.data, verbose=False)
	mfit2 = model.fit(data=res.data,
					  opt_algorithm='BFGS',
					  verbose=True)
	mfit.ae_metrics(actual=res.params)

	assert(mfit.name == 'MLModel')
	assert(mfit.method == 'Maximum Likelihood')
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

	taskresults = db.lr_cr(narms=2).simulate(nsubjects=nsubjects, ntrials=ntrials)
	banditgm = gm.bandit(model='lr_cr')

	model = MCMC(generative_model=banditgm)

	assert(model.name == 'FitrMCMCModel')
	assert(model.generative_model == banditgm)

	lrcr = model.fit(data=taskresults.data_mcmc, n_iterations=10)
	lrcr.trace_plot(save_figure=True)

	assert(lrcr.name == 'FitrMCMCModel')
	assert(lrcr.method == 'MCMC')
	assert(lrcr.nsubjects == 5)
	assert(lrcr.nparams == 2)
	assert(np.shape(lrcr.params) == (5, 2))
	assert(len(lrcr.paramnames) == 2)
	assert(type(lrcr.stanfit) == dict)

def test_fitrmodels():
	nsubjects = 5
	ntrials = 100

	lr = fitr.rlparams.LearningRate()
	cr = fitr.rlparams.ChoiceRandomness()
	params = [lr, cr]

	taskresults = db.lr_cr(narms=2).simulate(nsubjects=nsubjects, ntrials=ntrials)

	banditll = db.lr_cr(narms=2).loglikelihood
	banditgm = gm.bandit(model='lr_cr')

	model = FitModel(name='My 2-Armed Bandit Model',
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

	mlfit = model.fit(data=taskresults.data,
					  method='MLE',
					  verbose=False)
