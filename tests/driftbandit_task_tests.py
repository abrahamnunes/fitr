# -*- coding: utf-8 -*-

import numpy as np
from fitr.models import driftbandit as db

def test_bandit_lr_cr():
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

				lc_d = lc.simulate(nsubjects=N,
									 ntrials=T,
								     group_id='A',
							   	     preset_rpaths=preset,
								     rpath_common=common)
				lc_LL = lc.loglikelihood(params=[0.1, 4],
									     states=lc_d.data[0]['S'],
										 actions=lc_d.data[0]['A'],
										 rewards=lc_d.data[0]['R'])
				lc_CV = lc.loacv(params=[0.1, 4],
								 states=lc_d.data[0]['S'],
								 actions=lc_d.data[0]['A'],
								 rewards=lc_d.data[0]['R'])


				# Test lrcr_res
				assert(isinstance(lc_LL, np.float64))
				assert(lc.narms == k)
				assert(len(lc_d.data) == N)
				assert(lc_d.data_mcmc['N'] == N)
				assert(lc_d.data_mcmc['T'] == T)
				assert(np.shape(lc_d.data_mcmc['A']) == (T, N))
				assert(np.shape(lc_d.data_mcmc['R']) == (T, N))

				# Test loacv function output
				assert(isinstance(lc_CV, dict))
				assert(np.size(lc_CV['S']) == T)
				assert(np.size(lc_CV['A']) == T)
				assert(np.size(lc_CV['R']) == T)
				assert(np.size(lc_CV['A_pred']) == T)
				assert(np.size(lc_CV['A_match']) == T)
				assert(np.size(lc_CV['dLL']) == T)
				assert(np.size(lc_CV['dLL_pred']) == T)
				assert(isinstance(lc_CV['LL'], np.float64))
				assert(isinstance(lc_CV['LL_pred'], np.float64))
				assert(isinstance(lc_CV['acc'], np.float64))

def test_bandit_lr_cr_rs():
	N = 5
	T = 10

	# Loop over whether reward paths are common
	for common in [True, False]:
		# Loop over various numbers of arms
		for k in range(2, 7):
			preset_paths = np.random.uniform(0, 1, size=[T, k, N])

			# Loop over whether preset_paths are used
			for preset in [preset_paths, None]:
				lcr = db.lr_cr_rs(narms=k)

				lcr_d = lcr.simulate(nsubjects=N,
											 ntrials=T,
											 group_id='A',
											 preset_rpaths=preset,
											 rpath_common=common)
				lcr_LL = lcr.loglikelihood(params=[0.1, 4, 0.9],
										   states=lcr_d.data[0]['S'],
									       actions=lcr_d.data[0]['A'],
									       rewards=lcr_d.data[0]['R'])
				lcr_CV = lcr.loacv(params=[0.1, 4, 0.9],
								   states=lcr_d.data[0]['S'],
								   actions=lcr_d.data[0]['A'],
								   rewards=lcr_d.data[0]['R'])

				# Test lrcrrs_res
				assert(isinstance(lcr_LL, np.float64))
				assert(lcr.narms == k)
				assert(len(lcr_d.data) == N)
				assert(lcr_d.data_mcmc['N'] == N)
				assert(lcr_d.data_mcmc['T'] == T)
				assert(np.shape(lcr_d.data_mcmc['A']) == (T, N))
				assert(np.shape(lcr_d.data_mcmc['R']) == (T, N))

				# Test loacv function output
				assert(isinstance(lcr_CV, dict))
				assert(np.size(lcr_CV['S']) == T)
				assert(np.size(lcr_CV['A']) == T)
				assert(np.size(lcr_CV['R']) == T)
				assert(np.size(lcr_CV['A_pred']) == T)
				assert(np.size(lcr_CV['A_match']) == T)
				assert(np.size(lcr_CV['dLL']) == T)
				assert(np.size(lcr_CV['dLL_pred']) == T)
				assert(isinstance(lcr_CV['LL'], np.float64))
				assert(isinstance(lcr_CV['LL_pred'], np.float64))
				assert(isinstance(lcr_CV['acc'], np.float64))


def test_bandit_lr_cr_p():
	N = 5
	T = 10

	# Loop over whether reward paths are common
	for common in [True, False]:
		# Loop over various numbers of arms
		for k in range(2, 7):
			preset_paths = np.random.uniform(0, 1, size=[T, k, N])

			# Loop over whether preset_paths are used
			for preset in [preset_paths, None]:
				lcp = db.lr_cr_p(narms=k)

				lcp_d = lcp.simulate(nsubjects=N,
									 ntrials=T,
								     group_id='A',
							  	     preset_rpaths=preset,
								     rpath_common=common)
				lcp_LL = lcp.loglikelihood(params=[0.1, 4, 0.01],
										   states=lcp_d.data[0]['S'],
									       actions=lcp_d.data[0]['A'],
								     	   rewards=lcp_d.data[0]['R'])
				lcp_CV = lcp.loacv(params=[0.1, 4, 0.01],
								   states=lcp_d.data[0]['S'],
								   actions=lcp_d.data[0]['A'],
								   rewards=lcp_d.data[0]['R'])


				# Test lrcrp_res
				assert(isinstance(lcp_LL, np.float64))
				assert(lcp.narms == k)
				assert(len(lcp_d.data) == N)
				assert(lcp_d.data_mcmc['N'] == N)
				assert(lcp_d.data_mcmc['T'] == T)
				assert(np.shape(lcp_d.data_mcmc['A']) == (T, N))
				assert(np.shape(lcp_d.data_mcmc['R']) == (T, N))

				# Test loacv function output
				assert(isinstance(lcp_CV, dict))
				assert(np.size(lcp_CV['S']) == T)
				assert(np.size(lcp_CV['A']) == T)
				assert(np.size(lcp_CV['R']) == T)
				assert(np.size(lcp_CV['A_pred']) == T)
				assert(np.size(lcp_CV['A_match']) == T)
				assert(np.size(lcp_CV['dLL']) == T)
				assert(np.size(lcp_CV['dLL_pred']) == T)
				assert(isinstance(lcp_CV['LL'], np.float64))
				assert(isinstance(lcp_CV['LL_pred'], np.float64))
				assert(isinstance(lcp_CV['acc'], np.float64))

def test_bandit_lr_cr_rs_p():
	N = 5
	T = 10

	# Loop over whether reward paths are common
	for common in [True, False]:
		# Loop over various numbers of arms
		for k in range(2, 7):
			preset_paths = np.random.uniform(0, 1, size=[T, k, N])

			# Loop over whether preset_paths are used
			for preset in [preset_paths, None]:
				lcrp = db.lr_cr_rs_p(narms=k)

				lcrp_d = lcrp.simulate(nsubjects=N,
										 ntrials=T,
											   group_id='A',
											   preset_rpaths=preset,
											   rpath_common=common)
				lcrp_LL = lcrp.loglikelihood(params=[0.1, 4, 0.9, 0.01],
											 states=lcrp_d.data[0]['S'],
									         actions=lcrp_d.data[0]['A'],
							     		     rewards=lcrp_d.data[0]['R'])
				lcrp_CV = lcrp.loacv(params=[0.1, 4, 0.9, 0.01],
								     states=lcrp_d.data[0]['S'],
								     actions=lcrp_d.data[0]['A'],
								     rewards=lcrp_d.data[0]['R'])

				# Test lrcrrsp_res
				assert(isinstance(lcrp_LL, np.float64))
				assert(lcrp.narms == k)
				assert(len(lcrp_d.data) == N)
				assert(lcrp_d.data_mcmc['N'] == N)
				assert(lcrp_d.data_mcmc['T'] == T)
				assert(np.shape(lcrp_d.data_mcmc['A']) == (T, N))
				assert(np.shape(lcrp_d.data_mcmc['R']) == (T, N))

				# Test loacv function output
				assert(isinstance(lcrp_CV, dict))
				assert(np.size(lcrp_CV['S']) == T)
				assert(np.size(lcrp_CV['A']) == T)
				assert(np.size(lcrp_CV['R']) == T)
				assert(np.size(lcrp_CV['A_pred']) == T)
				assert(np.size(lcrp_CV['A_match']) == T)
				assert(np.size(lcrp_CV['dLL']) == T)
				assert(np.size(lcrp_CV['dLL_pred']) == T)
				assert(isinstance(lcrp_CV['LL'], np.float64))
				assert(isinstance(lcrp_CV['LL_pred'], np.float64))
				assert(isinstance(lcrp_CV['acc'], np.float64))

def test_bandit_dummy():
	N = 5
	T = 10

	# Loop over whether reward paths are common
	for common in [True, False]:
		# Loop over various numbers of arms
		for k in range(2, 7):
			preset_paths = np.random.uniform(0, 1, size=[T, k, N])

			# Loop over whether preset_paths are used
			for preset in [preset_paths, None]:
				dummy = db.dummy(narms=k)


				dummy_d = dummy.simulate(nsubjects=N,
										 ntrials=T,
									     group_id='A',
									     preset_rpaths=preset,
								         rpath_common=common)
				dummy_LL = dummy.loglikelihood(params=[4],
											   states=dummy_d.data[0]['S'],
											   actions=dummy_d.data[0]['A'],
											   rewards=dummy_d.data[0]['R'])
				dummy_CV = dummy.loacv(params=[4],
								       states=dummy_d.data[0]['S'],
								       actions=dummy_d.data[0]['A'],
								       rewards=dummy_d.data[0]['R'])

				# Test dummy_res
				assert(isinstance(dummy_LL, np.float64))
				assert(dummy.narms == k)
				assert(len(dummy_d.data) == N)
				assert(dummy_d.data_mcmc['N'] == N)
				assert(dummy_d.data_mcmc['T'] == T)
				assert(np.shape(dummy_d.data_mcmc['A']) == (T, N))
				assert(np.shape(dummy_d.data_mcmc['R']) == (T, N))

				# Test loacv function output
				assert(isinstance(dummy_CV, dict))
				assert(np.size(dummy_CV['S']) == T)
				assert(np.size(dummy_CV['A']) == T)
				assert(np.size(dummy_CV['R']) == T)
				assert(np.size(dummy_CV['A_pred']) == T)
				assert(np.size(dummy_CV['A_match']) == T)
				assert(np.size(dummy_CV['dLL']) == T)
				assert(np.size(dummy_CV['dLL_pred']) == T)
				assert(isinstance(dummy_CV['LL'], np.float64))
				assert(isinstance(dummy_CV['LL_pred'], np.float64))
				assert(isinstance(dummy_CV['acc'], np.float64))
