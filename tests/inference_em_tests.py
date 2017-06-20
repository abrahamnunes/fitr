# -*- coding: utf-8 -*-

import numpy as np

from fitr.rlparams import *
from fitr.inference import EM
from fitr.models import twostep as task

def test_em_vanilla():
	ntrials = 100
	nsubjects = 5

	res = task.lr_cr_mf().simulate(ntrials=ntrials, nsubjects=nsubjects)

	lr = LearningRate()
	cr = ChoiceRandomness()
	likfun = task.lr_cr_mf().loglikelihood

	model = EM(loglik_func=likfun,
			   params=[lr, cr],
			   name='EMModel')

	assert(model.name == 'EMModel')
	assert(len(model.params) == 2)
	assert(model.loglik_func == likfun)

	mfit = model.fit(data=res.data,
					 n_iterations=2,
					 early_stopping=False)

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
	assert(isinstance(mfit.ts_LME, list))
	assert(isinstance(mfit.ts_nLL, list))
	assert(isinstance(mfit.ts_BIC, list))
	assert(isinstance(mfit.ts_AIC, list))

def test_em_options():
	ntrials = 100
	nsubjects = 5

	res = task.lr_cr_mf().simulate(ntrials=ntrials, nsubjects=nsubjects)

	lr = LearningRate()
	cr = ChoiceRandomness()
	likfun = task.lr_cr_mf().loglikelihood

	model = EM(loglik_func=likfun,
			   params=[lr, cr],
			   name='EMModel')

	assert(model.name == 'EMModel')
	assert(len(model.params) == 2)
	assert(model.loglik_func == likfun)

	mfit = model.fit(data=res.data,
					 n_iterations=2,
					 opt_algorithm='BFGS',
					 init_grid=True,
					 grid_reinit=True,
					 n_grid_points=5,
					 n_reinit=1,
					 dofull=False,
					 early_stopping=True,
					 verbose=False)

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
	assert(isinstance(mfit.ts_LME, list))
	assert(isinstance(mfit.ts_nLL, list))
	assert(isinstance(mfit.ts_BIC, list))
	assert(isinstance(mfit.ts_AIC, list))
