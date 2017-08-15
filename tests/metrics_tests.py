# -*- coding: utf-8 -*-

from fitr.criticism import model_evaluation
from fitr.criticism.distance import parameter_distance
from fitr.criticism.distance import likelihood_distance
from fitr.models import twostep
import numpy as np

# Test that BIC is finite
def test_bic():
	assert(np.isfinite(model_evaluation.BIC(-10, 2, 10)))

# Test that AIC is finite
def test_aic():
	assert(np.isfinite(model_evaluation.AIC(2, -10)))

# Test that LME is finite
def test_lme():
	_testhess = np.array([[1, 0.7],[0.6, 1]])
	assert(np.isfinite(model_evaluation.LME(-10, 2, _testhess)))

# Test parameter_distance function
def test_parameter_distance():
	rand_params = np.random.uniform(0, 1, size=(10, 2))

	D = parameter_distance(params=rand_params,
						   dist_metric='canberra',
						   scale='minmax',
						   return_scaled=False)
	assert(np.shape(D) == (10, 10))

	D = parameter_distance(params=rand_params,
						   dist_metric='canberra',
						   scale='standard',
						   return_scaled=True)
	assert(len(D) == 2)

def test_loglikelihood_distance():
	ntrials = 10
	nsubjects = 10
	res = twostep.lr_cr_mf().simulate(ntrials=ntrials, nsubjects=nsubjects)

	loglik_fun = twostep.lr_cr_mf().loglikelihood
	for diffmetric, verbose in zip(['sq', 'diff', 'abs'], [True, False, False]):
		D = likelihood_distance(loglik_func=loglik_fun,
								data=res.data,
								params=res.params,
								diff_metric=diffmetric,
								verbose=verbose)

		assert(np.shape(D) == (10, 10))
