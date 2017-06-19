# -*- coding: utf-8 -*-

from fitr.rlparams import *
from fitr.inference import FitModel
from fitr.models import driftbandit as db

def test_fitrmodels():
	nsubjects = 5
	ntrials = 100

	lr = LearningRate()
	cr = ChoiceRandomness()
	params = [lr, cr]

	taskresults = db.lr_cr(narms=2).simulate(nsubjects=nsubjects, ntrials=ntrials)

	banditll = db.lr_cr(narms=2).loglikelihood
	banditgm = db.lr_cr(narms=2).gm

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
