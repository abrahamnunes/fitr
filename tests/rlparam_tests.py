# -*- coding: utf-8 -*-
import pytest
import fitr
import numpy as np

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

# Test that param specification errors run well
def test_paramspec_errors():
	prm = fitr.rlparams.LearningRate()
	with pytest.raises(Exception):
		fitr.rlparams.LearningRate(mean=0.5, sd=-1)
		fitr.rlparams.LearningRate(mean=0.5, sd=0)
		fitr.rlparams.LearningRate(mean=-0.5, sd=0.1)
		fitr.rlparams.LearningRate(mean=0, sd=0.1)
		fitr.rlparams.LearningRate(mean=1, sd=0.1)
		fitr.rlparams.LearningRate(mean=1.5, sd=0.1)
		fitr.rlparams.LearningRate(mean=0.4, sd=0.49)
		fitr.rlparams.ChoiceRandomness(mean=-1, sd=2)
		fitr.rlparams.ChoiceRandomness(mean=0, sd=2)

		prm.convert_meansd(mean=0.5, sd=0.1, dist='normal')
