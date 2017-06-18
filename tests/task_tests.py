# -*- coding: utf-8 -*-

import fitr
from fitr import tasks
from fitr.models import twostep
from fitr.models import driftbandit as db
import numpy as np
import scipy

def test_bandit():
	N = 5
	T = 10

	for k in range(2, 7):
		lrcr = db.lr_cr(narms=k)
		lrcrrs = db.lr_cr_rs(narms=k)
		lrcrp = db.lr_cr_p(narms=k)
		lrcrrsp = db.lr_cr_rs_p(narms=k)
		dummy = db.dummy(narms=k)

		lrcr_res = lrcr.simulate(nsubjects=N, ntrials=T)
		lrcrrs_res = lrcrrs.simulate(nsubjects=N, ntrials=T)
		lrcrp_res = lrcrp.simulate(nsubjects=N, ntrials=T)
		lrcrrsp_res = lrcrrsp.simulate(nsubjects=N, ntrials=T)
		dummy_res = dummy.simulate(nsubjects=N, ntrials=T)

		# Test lrcrrs_res
		assert(lrcr.narms == k)
		assert(len(lrcr_res.data) == N)
		assert(lrcr_res.data_mcmc['N'] == N)
		assert(lrcr_res.data_mcmc['T'] == T)
		assert(np.shape(lrcr_res.data_mcmc['A']) == (T, N))
		assert(np.shape(lrcr_res.data_mcmc['R']) == (T, N))

		# Test lrcrrs_res
		assert(lrcrrs.narms == k)
		assert(len(lrcrrs_res.data) == N)
		assert(lrcrrs_res.data_mcmc['N'] == N)
		assert(lrcrrs_res.data_mcmc['T'] == T)
		assert(np.shape(lrcrrs_res.data_mcmc['A']) == (T, N))
		assert(np.shape(lrcrrs_res.data_mcmc['R']) == (T, N))

		# Test lrcrp_res
		assert(lrcrp.narms == k)
		assert(len(lrcrp_res.data) == N)
		assert(lrcrp_res.data_mcmc['N'] == N)
		assert(lrcrp_res.data_mcmc['T'] == T)
		assert(np.shape(lrcrp_res.data_mcmc['A']) == (T, N))
		assert(np.shape(lrcrp_res.data_mcmc['R']) == (T, N))

		# Test lrcrrsp_res
		assert(lrcrrsp.narms == k)
		assert(len(lrcrrsp_res.data) == N)
		assert(lrcrrsp_res.data_mcmc['N'] == N)
		assert(lrcrrsp_res.data_mcmc['T'] == T)
		assert(np.shape(lrcrrsp_res.data_mcmc['A']) == (T, N))
		assert(np.shape(lrcrrsp_res.data_mcmc['R']) == (T, N))

		# Test dummy_res
		assert(dummy.narms == k)
		assert(len(dummy_res.data) == N)
		assert(dummy_res.data_mcmc['N'] == N)
		assert(dummy_res.data_mcmc['T'] == T)
		assert(np.shape(dummy_res.data_mcmc['A']) == (T, N))
		assert(np.shape(dummy_res.data_mcmc['R']) == (T, N))

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
