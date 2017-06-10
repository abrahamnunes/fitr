# -*- coding: utf-8 -*-

import fitr
from fitr import tasks
from fitr import generative_models as gm
from fitr import loglik_functions as ll
from fitr import model_selection
import numpy as np
import scipy

def test_model_selections():
	nsubjects = 5
	ntrials = 10

	lr = fitr.rlparams.LearningRate()
	cr = fitr.rlparams.ChoiceRandomness()
	params = [lr, cr]

	group = np.zeros([nsubjects, 2])
	group[:, 0] = lr.sample(size=nsubjects)
	group[:, 1] = cr.sample(size=nsubjects)

	res = tasks.bandit(narms=2).simulate(params=group, ntrials=ntrials)

	m1 = fitr.fitr.EM(loglik_func=ll.bandit_ll().lr_cr,
						 	params=params)

	m2 = fitr.fitr.EM(loglik_func=ll.bandit_ll().lr_cr_rs,
				 	  params=[fitr.rlparams.LearningRate(),
					  		  fitr.rlparams.ChoiceRandomness(),
							  fitr.rlparams.RewardSensitivity()])

	fit1 = m1.fit(data=res.data)
	fit2 = m2.fit(data=res.data)

	models = [fit1, fit2]
	bms_results = model_selection.BMS(model_fits=models, c_limit=1e-10).run()
	bic_results = model_selection.BIC(model_fits=models).run()
	aic_results = model_selection.AIC(model_fits=models).run()

	assert(len(bms_results.modelnames) == 2)
	assert(len(bms_results.xp) == 2)
	assert(len(bms_results.pxp) == 2)

	assert(len(bic_results.modelnames) == 2)
	assert(len(bic_results.BIC) == 2)

	assert(len(aic_results.modelnames) == 2)
	assert(len(aic_results.AIC) == 2)
