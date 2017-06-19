# -*- coding: utf-8 -*-

import numpy as np
from fitr.models import twostep as ts

def test_twostep_lr_cr_mf():
	nsubjects = 5
	ntrials = 10

	for rpath_common in [False, True]:
		preset_paths = np.random.uniform(0, 1, size=[ntrials, 4, nsubjects])
		for preset in [preset_paths, None]:
			res = ts.lr_cr_mf().simulate(ntrials=ntrials,
										 nsubjects=nsubjects,
										 group_id='A',
										 rpath_common=rpath_common,
									     preset_rpaths=preset)
			LL = ts.lr_cr_mf().loglikelihood(params=res.params[0],
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
			res = ts.lr_cr_rs_mf().simulate(ntrials=ntrials,
											nsubjects=nsubjects,
											group_id='A',
											rpath_common=rpath_common,
											preset_rpaths=preset)
			LL = ts.lr_cr_rs_mf().loglikelihood(params=res.params[0],
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
			res = ts.lr_cr_et_mf().simulate(ntrials=ntrials,
											nsubjects=nsubjects,
											group_id='A',
											rpath_common=rpath_common,
											preset_rpaths=preset)
			LL = ts.lr_cr_et_mf().loglikelihood(params=res.params[0],
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
			res = ts.lr_cr_p_mf().simulate(ntrials=ntrials,
										   nsubjects=nsubjects,
										   group_id='A',
										   rpath_common=rpath_common,
										   preset_rpaths=preset)
			LL = ts.lr_cr_p_mf().loglikelihood(params=res.params[0],
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
			res = ts.lr_cr_et_p_mf().simulate(ntrials=ntrials,
											  nsubjects=nsubjects,
											  group_id='A',
											  rpath_common=rpath_common,
											  preset_rpaths=preset)
			LL = ts.lr_cr_et_p_mf().loglikelihood(params=res.params[0],
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
			res = ts.lr_cr_rs_p_mf().simulate(ntrials=ntrials,
											  nsubjects=nsubjects,
										      group_id='A',
											  rpath_common=rpath_common,
											  preset_rpaths=preset)

			LL = ts.lr_cr_rs_p_mf().loglikelihood(params=res.params[0],
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
			res = ts.lr_cr_rs_et_p_mf().simulate(ntrials=ntrials,
												 nsubjects=nsubjects,
												 group_id='A',
												 rpath_common=rpath_common,
												 preset_rpaths=preset)
			LL = ts.lr_cr_rs_et_p_mf().loglikelihood(params=res.params[0],
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

def test_twostep_lr_cr_w():
	nsubjects = 5
	ntrials = 10

	for rpath_common in [False, True]:
		preset_paths = np.random.uniform(0, 1, size=[ntrials, 4, nsubjects])
		for preset in [preset_paths, None]:
			res = ts.lr_cr_w().simulate(ntrials=ntrials,
										nsubjects=nsubjects,
										group_id='A',
									    rpath_common=rpath_common,
										preset_rpaths=preset)
			LL = ts.lr_cr_w().loglikelihood(params=res.params[0],
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

def test_twostep_lr_cr_et_w():
	nsubjects = 5
	ntrials = 10

	for rpath_common in [False, True]:
		preset_paths = np.random.uniform(0, 1, size=[ntrials, 4, nsubjects])
		for preset in [preset_paths, None]:
			res = ts.lr_cr_et_w().simulate(ntrials=ntrials,
										   nsubjects=nsubjects,
										   group_id='A',
									       rpath_common=rpath_common,
										   preset_rpaths=preset)
			LL = ts.lr_cr_et_w().loglikelihood(params=res.params[0],
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

def test_twostep_lr_cr_p_w():
	nsubjects = 5
	ntrials = 10

	for rpath_common in [False, True]:
		preset_paths = np.random.uniform(0, 1, size=[ntrials, 4, nsubjects])
		for preset in [preset_paths, None]:
			res = ts.lr_cr_p_w().simulate(ntrials=ntrials,
										  nsubjects=nsubjects,
										  group_id='A',
									      rpath_common=rpath_common,
										  preset_rpaths=preset)
			LL = ts.lr_cr_p_w().loglikelihood(params=res.params[0],
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

def test_twostep_lr_cr_et_p_w():
	nsubjects = 5
	ntrials = 10

	for rpath_common in [False, True]:
		preset_paths = np.random.uniform(0, 1, size=[ntrials, 4, nsubjects])
		for preset in [preset_paths, None]:
			res = ts.lr_cr_et_p_w().simulate(ntrials=ntrials,
											 nsubjects=nsubjects,
											 group_id='A',
									         rpath_common=rpath_common,
											 preset_rpaths=preset)
			LL = ts.lr_cr_et_p_w().loglikelihood(params=res.params[0],
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

def test_twostep_lr_cr_rs_et_p_w():
	nsubjects = 5
	ntrials = 10

	for rpath_common in [False, True]:
		preset_paths = np.random.uniform(0, 1, size=[ntrials, 4, nsubjects])
		for preset in [preset_paths, None]:
			res = ts.lr_cr_rs_et_p_w().simulate(ntrials=ntrials,
											    nsubjects=nsubjects,
											    group_id='A',
									    		rpath_common=rpath_common,
											    preset_rpaths=preset)
			LL = ts.lr_cr_rs_et_p_w().loglikelihood(params=res.params[0],
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
