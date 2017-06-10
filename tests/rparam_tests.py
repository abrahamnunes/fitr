# -*- coding: utf-8 -*-

import fitr
import numpy as np
import scipy

# Test that Param object can be instantiated appropriately
def test_paramobj():
	par = fitr.rlparams.Param(name='Testname', rng='unit')
	assert(par.name == 'Testname')
	assert(par.rng == 'unit')
	assert(par.dist is None)

# Test that other parameters sample appropriately
def test_paramsampling():
	lr = fitr.rlparams.LearningRate(mean=0.5, sd=0.1)
	rs = fitr.rlparams.RewardSensitivity(mean=0.5, sd=0.1)
	et = fitr.rlparams.EligibilityTrace(mean=0.5, sd=0.1)
	w = fitr.rlparams.MBMF_Balance(mean=0.5, sd=0.1)
	cr = fitr.rlparams.ChoiceRandomness(mean=5, sd=1)
	ps = fitr.rlparams.Perseveration(mean=0, sd=1)

	nsubj = 10
	assert(np.size(lr.sample(size=nsubj)) == nsubj)
	assert(np.size(rs.sample(size=nsubj)) == nsubj)
	assert(np.size(et.sample(size=nsubj)) == nsubj)
	assert(np.size(w.sample(size=nsubj)) == nsubj)
	assert(np.size(cr.sample(size=nsubj)) == nsubj)
	assert(np.size(ps.sample(size=nsubj)) == nsubj)
