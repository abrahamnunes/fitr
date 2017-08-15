# -*- coding: utf-8 -*-

import numpy as np
from fitr.criticism.ae_stats import paramcorr
from fitr.criticism.ae_stats import param_ttest

def test_paramcorr():
	n_params = 3
	n_subjects = 3

	x = np.random.normal(0, 1, size=(n_subjects, n_params))
	y = np.random.normal(0, 1, size=(n_subjects, n_params))

	corrs = paramcorr(X=x, Y=y)
	assert(np.shape(corrs) == (n_params, n_sujects))

def test_param_ttest():
	n_params = 3
	n_subjects = 30

	x = np.random.normal(0, 1, size=(n_subjects, n_params))
	y = np.random.normal(0, 1, size=(n_subjects, n_params))

	res = param_ttest(X=x, Y=y)
	assert(np.shape(res) == (n_params, n_sujects))
