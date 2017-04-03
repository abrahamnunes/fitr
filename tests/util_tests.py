# -*- coding: utf-8 -*-

import fitr
import numpy as np
import scipy

# Test softmax is returning numpy array
# Test softmax returns array with correct dimensional length
def test_softmax():
	_temp = fitr.utils.softmax([0.1, 0.2])
	assert(type(_temp) is np.ndarray)
	assert(np.shape(_temp)[0] == 2)

# Test logsumexp is of type numpy.float64
# Test that results are the same as the scipy method
def test_logsumexp():
    _temp = fitr.utils.logsumexp([0.1, 0.2, 100, 200, 500])
    assert(type(_temp) is np.float64)
    assert(round(_temp) == round(scipy.misc.logsumexp([0.1, 0.2, 100, 200, 500])))
