# -*- coding: utf-8 -*-

import fitr
from fitr import tasks
import numpy as np
import scipy

def test_bandit():
	lr = fitr.rlparams.LearningRate()
	cr = fitr.rlparams.ChoiceRandomness()
	group = fitr.rlparams.generate_group(params=[lr, cr], nsubjects=10)
	task = tasks.bandit()
	res = task.simulate(ntrials=10, params=group)

	assert(res.params.all() == group.all())
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
	task = tasks.twostep()
	res = task.simulate(ntrials=5, params=group)

	assert(res.params.all() == group.all())
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

def test_action():
	assert(type(tasks.action(x=np.array([0.8, 0.2]))) == np.int64)

def test_reward():
	paths = np.array([0.25, 0.25, 0.25])
	assert(np.isfinite(tasks.reward(a=1, paths=paths)))
