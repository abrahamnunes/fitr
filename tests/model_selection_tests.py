# -*- coding: utf-8 -*-

from fitr.rlparams import LearningRate
from fitr.rlparams import ChoiceRandomness
from fitr.rlparams import RewardSensitivity
from fitr.inference import EM
from fitr.criticism.model_selection import BIC
from fitr.criticism.model_selection import AIC
from fitr.criticism.model_selection import BMS

from fitr.models import driftbandit

def test_model_selections():
	nsubjects = 5
	ntrials = 10

	lr = LearningRate()
	cr = ChoiceRandomness()
	params = [lr, cr]

	task = driftbandit.lr_cr(narms=2)
	res = task.simulate(nsubjects=nsubjects, ntrials=ntrials)

	m1 = EM(loglik_func=task.loglikelihood,
			params=params)

	m2 = EM(loglik_func=driftbandit.lr_cr_rs(narms=2).loglikelihood,
			params=[LearningRate(),
		  		    ChoiceRandomness(),
				    RewardSensitivity()])

	fit1 = m1.fit(data=res.data)
	fit2 = m2.fit(data=res.data)

	models = [fit1, fit2]
	bms_results = BMS(model_fits=models, c_limit=1e-10).run()
	bms_results.plot(statistic='pxp', save_figure=True)
	bms_results.plot(statistic='xp')

	bic_results = BIC(model_fits=models).run()
	bic_results.plot(statistic='BIC')

	aic_results = AIC(model_fits=models).run()
	aic_results.plot(statistic='AIC')

	assert(len(bms_results.modelnames) == 2)
	assert(len(bms_results.xp) == 2)
	assert(len(bms_results.pxp) == 2)

	assert(len(bic_results.modelnames) == 2)
	assert(len(bic_results.BIC) == 2)

	assert(len(aic_results.modelnames) == 2)
	assert(len(aic_results.AIC) == 2)
