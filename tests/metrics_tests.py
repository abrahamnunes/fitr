# -*- coding: utf-8 -*-

import fitr
import numpy as np

# Test that BIC is finite
def test_bic():
	assert(np.isfinite(fitr.metrics.BIC(-10, 2, 10)))

# Test that AIC is finite
def test_aic():
	assert(np.isfinite(fitr.metrics.AIC(2, -10)))

# Test that LME is finite
def test_lme():
	_testhess = np.array([[1, 0.7],[0.6, 1]])
	assert(np.isfinite(fitr.metrics.LME(-10, 2, _testhess)))
