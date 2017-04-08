# -*- coding: utf-8 -*-

import fitr
from fitr import tasks
from fitr import generative_models as gm
import numpy as np
import scipy

def test_gm():
	model = gm.GenerativeModel()
	assert(type(model.paramnames['long']) == list)
	assert(type(model.paramnames['code']) == list)
	assert(type(model.model) == str)

def test_banditgm_lrcr():
	model = gm.bandit(model='lr_cr')
	assert(type(model.paramnames['long']) == list)
	assert(type(model.paramnames['code']) == list)
	assert(len(model.paramnames['long']) == 2)
	assert(len(model.paramnames['code']) == 2)
	assert(type(model.model) == str)

def test_banditgm_lrcrrs():
	model = gm.bandit(model='lr_cr_rs')
	assert(type(model.paramnames['long']) == list)
	assert(type(model.paramnames['code']) == list)
	assert(len(model.paramnames['long']) == 3)
	assert(len(model.paramnames['code']) == 3)
	assert(type(model.model) == str)

def test_twostep_gm_lrcrw():
	model = gm.twostep(model='lr_cr_w')
	assert(type(model.paramnames['long']) == list)
	assert(type(model.paramnames['code']) == list)
	assert(len(model.paramnames['long']) == 3)
	assert(len(model.paramnames['code']) == 3)
	assert(type(model.model) == str)

def test_twostep_gm_lrcretw():
	model = gm.twostep(model='lr_cr_et_w')
	assert(type(model.paramnames['long']) == list)
	assert(type(model.paramnames['code']) == list)
	assert(len(model.paramnames['long']) == 4)
	assert(len(model.paramnames['code']) == 4)
	assert(type(model.model) == str)

def test_twostep_gm_lrcrwp():
	model = gm.twostep(model='lr_cr_w_p')
	assert(type(model.paramnames['long']) == list)
	assert(type(model.paramnames['code']) == list)
	assert(len(model.paramnames['long']) == 4)
	assert(len(model.paramnames['code']) == 4)
	assert(type(model.model) == str)

def test_twostep_gm_lrcretwp():
	model = gm.twostep(model='lr_cr_et_w_p')
	assert(type(model.paramnames['long']) == list)
	assert(type(model.paramnames['code']) == list)
	assert(len(model.paramnames['long']) == 5)
	assert(len(model.paramnames['code']) == 5)
	assert(type(model.model) == str)
