# -*- coding: utf-8 -*-

import fitr
from fitr import tasks
import fitr.twostep as twostep
import numpy as np
import scipy

def test_bandit():
	nsubjects = 5
	ntrials = 10

	lr = fitr.rlparams.LearningRate()
	cr = fitr.rlparams.ChoiceRandomness()

	group = np.zeros([nsubjects, 2])
	group[:, 0] = lr.sample(size=nsubjects)
	group[:, 1] = cr.sample(size=nsubjects)

	task = tasks.bandit()
	res = task.simulate(ntrials=ntrials, params=group)

	assert(res.params.all() == group.all())
	assert(len(res.data) == nsubjects)
	assert(res.data_mcmc['N'] == nsubjects)
	assert(res.data_mcmc['T'] == ntrials)
	assert(np.shape(res.data_mcmc['A']) == (ntrials, nsubjects))
	assert(np.shape(res.data_mcmc['R']) == (ntrials, nsubjects))

def test_twostep_lr_cr_mf():
	nsubjects = 5
	ntrials = 10

	for rpath_common in [False, True]:
		res = twostep.lr_cr_mf().simulate(ntrials=ntrials,
										  nsubjects=nsubjects,
										  rpath_common=rpath_common)
		LL = twostep.lr_cr_mf().loglikelihood(params=res.params[0],
											  states=res.data[0]['S'],
											  actions=res.data[0]['A'],
											  rewards=res.data[0]['R'])
		assert(type(LL) is np.float64)
		assert(len(res.data) == nsubjects)
		assert(res.data_mcmc['N'] == nsubjects)
		assert(res.data_mcmc['T'] == ntrials)
		assert(np.shape(res.data_mcmc['A1']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['A2']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['S2']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['R']) == (ntrials, nsubjects))

def test_twostep_lr_cr_rs_mf():
	nsubjects = 5
	ntrials = 10

	for rpath_common in [False, True]:
		res = twostep.lr_cr_rs_mf().simulate(ntrials=ntrials,
											 nsubjects=nsubjects,
											 rpath_common=rpath_common)
		LL = twostep.lr_cr_rs_mf().loglikelihood(params=res.params[0],
											  	 states=res.data[0]['S'],
											  	 actions=res.data[0]['A'],
											  	 rewards=res.data[0]['R'])
		assert(type(LL) is np.float64)
		assert(len(res.data) == nsubjects)
		assert(res.data_mcmc['N'] == nsubjects)
		assert(res.data_mcmc['T'] == ntrials)
		assert(np.shape(res.data_mcmc['A1']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['A2']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['S2']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['R']) == (ntrials, nsubjects))

def test_twostep_lr_cr_et_mf():
	nsubjects = 5
	ntrials = 10

	for rpath_common in [False, True]:
		res = twostep.lr_cr_et_mf().simulate(ntrials=ntrials,
											 nsubjects=nsubjects,
											 rpath_common=rpath_common)
		LL = twostep.lr_cr_et_mf().loglikelihood(params=res.params[0],
												 states=res.data[0]['S'],
												 actions=res.data[0]['A'],
												 rewards=res.data[0]['R'])
		assert(type(LL) is np.float64)
		assert(len(res.data) == nsubjects)
		assert(res.data_mcmc['N'] == nsubjects)
		assert(res.data_mcmc['T'] == ntrials)
		assert(np.shape(res.data_mcmc['A1']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['A2']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['S2']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['R']) == (ntrials, nsubjects))

def test_twostep_lr_cr_p_mf():
	nsubjects = 5
	ntrials = 10

	for rpath_common in [False, True]:
		res = twostep.lr_cr_p_mf().simulate(ntrials=ntrials,
											nsubjects=nsubjects,
											rpath_common=rpath_common)
		LL = twostep.lr_cr_p_mf().loglikelihood(params=res.params[0],
											  	states=res.data[0]['S'],
											  	actions=res.data[0]['A'],
											  	rewards=res.data[0]['R'])
		assert(type(LL) is np.float64)
		assert(len(res.data) == nsubjects)
		assert(res.data_mcmc['N'] == nsubjects)
		assert(res.data_mcmc['T'] == ntrials)
		assert(np.shape(res.data_mcmc['A1']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['A2']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['S2']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['R']) == (ntrials, nsubjects))

def test_twostep_lr_cr_et_p_mf():
	nsubjects = 5
	ntrials = 10

	for rpath_common in [False, True]:
		res = twostep.lr_cr_et_p_mf().simulate(ntrials=ntrials,
											   nsubjects=nsubjects,
											   rpath_common=rpath_common)
		LL = twostep.lr_cr_et_p_mf().loglikelihood(params=res.params[0],
											  	   states=res.data[0]['S'],
											  	   actions=res.data[0]['A'],
											  	   rewards=res.data[0]['R'])
		assert(type(LL) is np.float64)
		assert(len(res.data) == nsubjects)
		assert(res.data_mcmc['N'] == nsubjects)
		assert(res.data_mcmc['T'] == ntrials)
		assert(np.shape(res.data_mcmc['A1']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['A2']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['S2']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['R']) == (ntrials, nsubjects))

def test_twostep_lr_cr_rs_p_mf():
	nsubjects = 5
	ntrials = 10

	for rpath_common in [False, True]:
		res = twostep.lr_cr_rs_p_mf().simulate(ntrials=ntrials,
											   nsubjects=nsubjects,
											   rpath_common=rpath_common)

		LL = twostep.lr_cr_rs_p_mf().loglikelihood(params=res.params[0],
											  	   states=res.data[0]['S'],
											  	   actions=res.data[0]['A'],
											  	   rewards=res.data[0]['R'])
		assert(type(LL) is np.float64)
		assert(len(res.data) == nsubjects)
		assert(res.data_mcmc['N'] == nsubjects)
		assert(res.data_mcmc['T'] == ntrials)
		assert(np.shape(res.data_mcmc['A1']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['A2']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['S2']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['R']) == (ntrials, nsubjects))

def test_twostep_lr_cr_rs_et_p_mf():
	nsubjects = 5
	ntrials = 10

	for rpath_common in [False, True]:
		res = twostep.lr_cr_rs_et_p_mf().simulate(ntrials=ntrials,
												  nsubjects=nsubjects,
												  rpath_common=rpath_common)
		LL = twostep.lr_cr_rs_et_p_mf().loglikelihood(params=res.params[0],
											  	      states=res.data[0]['S'],
											  	      actions=res.data[0]['A'],
											  	      rewards=res.data[0]['R'])

		assert(type(LL) is np.float64)
		assert(len(res.data) == nsubjects)
		assert(res.data_mcmc['N'] == nsubjects)
		assert(res.data_mcmc['T'] == ntrials)
		assert(np.shape(res.data_mcmc['A1']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['A2']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['S2']) == (ntrials, nsubjects))
		assert(np.shape(res.data_mcmc['R']) == (ntrials, nsubjects))
