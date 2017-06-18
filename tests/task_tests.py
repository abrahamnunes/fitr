# -*- coding: utf-8 -*-

import fitr
from fitr.models import twostep
from fitr.models import driftbandit as db
import numpy as np
import scipy

def test_bandit():
	N = 5
	T = 10

	# Loop over whether reward paths are common
	for common in [True, False]:
		# Loop over various numbers of arms
		for k in range(2, 7):
			preset_paths = np.random.uniform(0, 1, size=[T, k, N])

			# Loop over whether preset_paths are used
			for preset in [preset_paths, None]:
				lc = db.lr_cr(narms=k)
				lcr = db.lr_cr_rs(narms=k)
				lcp = db.lr_cr_p(narms=k)
				lcrp = db.lr_cr_rs_p(narms=k)
				dummy = db.dummy(narms=k)

				lc_d = lc.simulate(nsubjects=N,
									 ntrials=T,
								     group_id='A',
							   	     preset_rpaths=preset,
								     rpath_common=common)
				lc_LL = lc.loglikelihood(params=[0.1, 4],
									     states=lc_d.data[0]['S'],
										 actions=lc_d.data[0]['A'],
										 rewards=lc_d.data[0]['R'])

				lcr_d = lcr.simulate(nsubjects=N,
											 ntrials=T,
											 group_id='A',
											 preset_rpaths=preset,
											 rpath_common=common)
				lcr_LL = lcr.loglikelihood(params=[0.1, 4, 0.9],
										   states=lcr_d.data[0]['S'],
									       actions=lcr_d.data[0]['A'],
									       rewards=lcr_d.data[0]['R'])

				lcp_d = lcp.simulate(nsubjects=N,
									 ntrials=T,
								     group_id='A',
							  	     preset_rpaths=preset,
								     rpath_common=common)
				lcp_LL = lcp.loglikelihood(params=[0.1, 4, 0.01],
										   states=lcp_d.data[0]['S'],
									       actions=lcp_d.data[0]['A'],
								     	   rewards=lcp_d.data[0]['R'])

				lcrp_d = lcrp.simulate(nsubjects=N,
										 ntrials=T,
											   group_id='A',
											   preset_rpaths=preset,
											   rpath_common=common)
				lcrp_LL = lcrp.loglikelihood(params=[0.1, 4, 0.9, 0.01],
											 states=lcrp_d.data[0]['S'],
									         actions=lcrp_d.data[0]['A'],
							     		     rewards=lcrp_d.data[0]['R'])

				dummy_d = dummy.simulate(nsubjects=N,
										 ntrials=T,
									     group_id='A',
									     preset_rpaths=preset,
								         rpath_common=common)
				dummy_LL = dummy.loglikelihood(params=[4],
											   states=dummy_d.data[0]['S'],
											   actions=dummy_d.data[0]['A'],
											   rewards=dummy_d.data[0]['R'])

				# Test lrcrrs_res
				assert(type(lc_LL) is np.float64)
				assert(lc.narms == k)
				assert(len(lc_d.data) == N)
				assert(lc_d.data_mcmc['N'] == N)
				assert(lc_d.data_mcmc['T'] == T)
				assert(np.shape(lc_d.data_mcmc['A']) == (T, N))
				assert(np.shape(lc_d.data_mcmc['R']) == (T, N))

				# Test lrcrrs_res
				assert(type(lcr_LL) is np.float64)
				assert(lcr.narms == k)
				assert(len(lcr_d.data) == N)
				assert(lcr_d.data_mcmc['N'] == N)
				assert(lcr_d.data_mcmc['T'] == T)
				assert(np.shape(lcr_d.data_mcmc['A']) == (T, N))
				assert(np.shape(lcr_d.data_mcmc['R']) == (T, N))

				# Test lrcrp_res
				assert(type(lcp_LL) is np.float64)
				assert(lcp.narms == k)
				assert(len(lcp_d.data) == N)
				assert(lcp_d.data_mcmc['N'] == N)
				assert(lcp_d.data_mcmc['T'] == T)
				assert(np.shape(lcp_d.data_mcmc['A']) == (T, N))
				assert(np.shape(lcp_d.data_mcmc['R']) == (T, N))

				# Test lrcrrsp_res
				assert(type(lcrp_LL) is np.float64)
				assert(lcrp.narms == k)
				assert(len(lcrp_d.data) == N)
				assert(lcrp_d.data_mcmc['N'] == N)
				assert(lcrp_d.data_mcmc['T'] == T)
				assert(np.shape(lcrp_d.data_mcmc['A']) == (T, N))
				assert(np.shape(lcrp_d.data_mcmc['R']) == (T, N))

				# Test dummy_res
				assert(type(dummy_LL) is np.float64)
				assert(dummy.narms == k)
				assert(len(dummy_d.data) == N)
				assert(dummy_d.data_mcmc['N'] == N)
				assert(dummy_d.data_mcmc['T'] == T)
				assert(np.shape(dummy_d.data_mcmc['A']) == (T, N))
				assert(np.shape(dummy_d.data_mcmc['R']) == (T, N))

def test_twostep_lr_cr_mf():
	nsubjects = 5
	ntrials = 10

	for rpath_common in [False, True]:
		preset_paths = np.random.uniform(0, 1, size=[ntrials, 4, nsubjects])
		for preset in [preset_paths, None]:
			res = twostep.lr_cr_mf().simulate(ntrials=ntrials,
											  nsubjects=nsubjects,
											  group_id='A',
											  rpath_common=rpath_common,
											  preset_rpaths=preset)
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
		preset_paths = np.random.uniform(0, 1, size=[ntrials, 4, nsubjects])
		for preset in [preset_paths, None]:
			res = twostep.lr_cr_rs_mf().simulate(ntrials=ntrials,
												 nsubjects=nsubjects,
												 group_id='A',
												 rpath_common=rpath_common,
												 preset_rpaths=preset)
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
		preset_paths = np.random.uniform(0, 1, size=[ntrials, 4, nsubjects])
		for preset in [preset_paths, None]:
			res = twostep.lr_cr_et_mf().simulate(ntrials=ntrials,
												 nsubjects=nsubjects,
												 group_id='A',
												 rpath_common=rpath_common,
												 preset_rpaths=preset)
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
		preset_paths = np.random.uniform(0, 1, size=[ntrials, 4, nsubjects])
		for preset in [preset_paths, None]:
			res = twostep.lr_cr_p_mf().simulate(ntrials=ntrials,
												nsubjects=nsubjects,
												group_id='A',
												rpath_common=rpath_common,
												preset_rpaths=preset)
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
		preset_paths = np.random.uniform(0, 1, size=[ntrials, 4, nsubjects])
		for preset in [preset_paths, None]:
			res = twostep.lr_cr_et_p_mf().simulate(ntrials=ntrials,
												   nsubjects=nsubjects,
												   group_id='A',
												   rpath_common=rpath_common,
												   preset_rpaths=preset)
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
		preset_paths = np.random.uniform(0, 1, size=[ntrials, 4, nsubjects])
		for preset in [preset_paths, None]:
			res = twostep.lr_cr_rs_p_mf().simulate(ntrials=ntrials,
												   nsubjects=nsubjects,
												   group_id='A',
												   rpath_common=rpath_common,
												   preset_rpaths=preset)

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
		preset_paths = np.random.uniform(0, 1, size=[ntrials, 4, nsubjects])
		for preset in [preset_paths, None]:
			res = twostep.lr_cr_rs_et_p_mf().simulate(ntrials=ntrials,
													  nsubjects=nsubjects,
													  group_id='A',
													  rpath_common=rpath_common,preset_rpaths=preset)
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
