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
	res = bandit_task.simulate(ntrials==10, params=group)

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
