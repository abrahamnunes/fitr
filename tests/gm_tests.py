# -*- coding: utf-8 -*-

import numpy as np
import scipy

import fitr
from fitr.models import driftbandit
from fitr.models import twostep
from fitr.models import GenerativeModel

from fitr.inference import MCMC

def test_gm():
	model = GenerativeModel()
	assert(type(model.paramnames['long']) == list)
	assert(type(model.paramnames['code']) == list)
	assert(type(model.model) == str)

def test_banditgm_lrcr():
	nsubjects = 10
	ntrials = 100
	narms = 2
	model = driftbandit.lr_cr(narms=narms)
	res = model.simulate(nsubjects=nsubjects, ntrials=ntrials)

	assert(type(model.gm.paramnames['long']) == list)
	assert(type(model.gm.paramnames['code']) == list)
	assert(len(model.gm.paramnames['long']) == 2)
	assert(len(model.gm.paramnames['code']) == 2)
	assert(type(model.gm.model) == str)
	assert(res.data_mcmc['K'] == narms)
	assert(res.data_mcmc['N'] == nsubjects)
	assert(res.data_mcmc['T'] == ntrials)

def test_banditgm_lrcrrs():
	nsubjects = 10
	ntrials = 100
	narms = 2

	model = driftbandit.lr_cr_rs(narms=narms)
	res = model.simulate(nsubjects=nsubjects, ntrials=ntrials)


	assert(type(model.gm.paramnames['long']) == list)
	assert(type(model.gm.paramnames['code']) == list)
	assert(len(model.gm.paramnames['long']) == 3)
	assert(len(model.gm.paramnames['code']) == 3)
	assert(type(model.gm.model) == str)
	assert(res.data_mcmc['K'] == narms)
	assert(res.data_mcmc['N'] == nsubjects)
	assert(res.data_mcmc['T'] == ntrials)

def test_banditgm_lrcrp():
	nsubjects = 10
	ntrials = 100
	narms = 2

	model = driftbandit.lr_cr_p(narms=narms)
	res = model.simulate(nsubjects=nsubjects, ntrials=ntrials)


	assert(type(model.gm.paramnames['long']) == list)
	assert(type(model.gm.paramnames['code']) == list)
	assert(len(model.gm.paramnames['long']) == 3)
	assert(len(model.gm.paramnames['code']) == 3)
	assert(type(model.gm.model) == str)
	assert(res.data_mcmc['K'] == narms)
	assert(res.data_mcmc['N'] == nsubjects)
	assert(res.data_mcmc['T'] == ntrials)

def test_banditgm_lrcrrsp():
	nsubjects = 10
	ntrials = 100
	narms = 2

	model = driftbandit.lr_cr_rs_p(narms=narms)
	res = model.simulate(nsubjects=nsubjects, ntrials=ntrials)


	assert(type(model.gm.paramnames['long']) == list)
	assert(type(model.gm.paramnames['code']) == list)
	assert(len(model.gm.paramnames['long']) == 4)
	assert(len(model.gm.paramnames['code']) == 4)
	assert(type(model.gm.model) == str)
	assert(res.data_mcmc['K'] == narms)
	assert(res.data_mcmc['N'] == nsubjects)
	assert(res.data_mcmc['T'] == ntrials)

def test_twostep_gm_lrcrw():
	nsubjects = 10
	ntrials = 100

	model = twostep.lr_cr_w()
	res = model.simulate(nsubjects=nsubjects, ntrials=ntrials)

	assert(type(model.gm.paramnames['long']) == list)
	assert(type(model.gm.paramnames['code']) == list)
	assert(len(model.gm.paramnames['long']) == 3)
	assert(len(model.gm.paramnames['code']) == 3)
	assert(type(model.gm.model) == str)
	assert(res.data_mcmc['N'] == nsubjects)
	assert(res.data_mcmc['T'] == ntrials)

def test_twostep_gm_lrcretw():
	nsubjects = 10
	ntrials = 100

	model = twostep.lr_cr_et_w()
	res = model.simulate(nsubjects=nsubjects, ntrials=ntrials)

	assert(type(model.gm.paramnames['long']) == list)
	assert(type(model.gm.paramnames['code']) == list)
	assert(len(model.gm.paramnames['long']) == 4)
	assert(len(model.gm.paramnames['code']) == 4)
	assert(type(model.gm.model) == str)
	assert(res.data_mcmc['N'] == nsubjects)
	assert(res.data_mcmc['T'] == ntrials)

def test_twostep_gm_lrcrpw():
	nsubjects = 10
	ntrials = 100

	model = twostep.lr_cr_p_w()
	res = model.simulate(nsubjects=nsubjects, ntrials=ntrials)

	assert(type(model.gm.paramnames['long']) == list)
	assert(type(model.gm.paramnames['code']) == list)
	assert(len(model.gm.paramnames['long']) == 4)
	assert(len(model.gm.paramnames['code']) == 4)
	assert(type(model.gm.model) == str)
	assert(res.data_mcmc['N'] == nsubjects)
	assert(res.data_mcmc['T'] == ntrials)

def test_twostep_gm_lrcretpw():
	nsubjects = 10
	ntrials = 100

	model = twostep.lr_cr_et_p_w()
	res = model.simulate(nsubjects=nsubjects, ntrials=ntrials)

	assert(type(model.gm.paramnames['long']) == list)
	assert(type(model.gm.paramnames['code']) == list)
	assert(len(model.gm.paramnames['long']) == 5)
	assert(len(model.gm.paramnames['code']) == 5)
	assert(type(model.gm.model) == str)
	assert(res.data_mcmc['N'] == nsubjects)
	assert(res.data_mcmc['T'] == ntrials)
