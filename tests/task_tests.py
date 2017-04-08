# -*- coding: utf-8 -*-

import fitr
import numpy as np
import scipy

def test_bandit():
	lr = fitr.rlparams.LearningRate()
	cr = fitr.rlparams.ChoiceRandomness()
	group = fitr.rlparams.generate_group(params=[lr, cr], nsubjects=10)
	task = fitr.tasks.bandit()
	res = task.simulate(ntrials=10, nsubjects=10)

	res.cumreward_param_plot()
	res.plot_cumreward()

	assert(len(res.params) == 2)
	assert(len(res.data) == 10)
	assert(res.data_mcmc['N'] == 10)
	assert(res.data_mcmc['T'] == 10)
	assert(np.shape(res.data_mcmc['A'])[0] == 10)
	assert(np.shape(res.data_mcmc['A'])[1] == 10)
	assert(np.shape(res.data_mcmc['R'])[0] == 10)
	assert(np.shape(res.data_mcmc['R'])[1] == 10)

def test_twostep():
	lr = fitr.rlparams.LearningRate()
	cr = fitr.rlparams.ChoiceRandomness()
	w = fitr.rlparams.MBMF_Balance()
	group = fitr.rlparams.generate_group(params=[lr, cr, w], nsubjects=5)
	task = fitr.tasks.twostep()
	res = task.simulate(ntrials=5, nsubjects=5)

	assert(len(res.params) == 3)
	assert(len(res.data) == 5)
	assert(res.data_mcmc['N'] == 5)
	assert(res.data_mcmc['T'] == 5)
	assert(np.shape(res.data_mcmc['A1'])[0] == 5)
	assert(np.shape(res.data_mcmc['A1'])[1] == 5)
	assert(np.shape(res.data_mcmc['A2'])[0] == 5)
	assert(np.shape(res.data_mcmc['A2'])[1] == 5)
	assert(np.shape(res.data_mcmc['S2'])[0] == 5)
	assert(np.shape(res.data_mcmc['S2'])[1] == 5)
	assert(np.shape(res.data_mcmc['R'])[0] == 5)
	assert(np.shape(res.data_mcmc['R'])[1] == 5)
