# -*- coding: utf-8 -*-

import numpy as np

from fitr.rlparams import *
from fitr.inference import EmpiricalPriors
from fitr.models import twostep as task

def test_empirical_priors():
	ntrials = 100
	nsubjects = 5

	lr = LearningRate()
	cr = ChoiceRandomness()

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
	assert(isinstance(mfit.ts_LME, list))
	assert(isinstance(mfit.ts_nLL, list))
	assert(isinstance(mfit.ts_BIC, list))
	assert(isinstance(mfit.ts_AIC, list))
