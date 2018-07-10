import numpy as np
from fitr.inference import mlepar
import pytest

def bad_loglik(x, D):
    i = x[0]
    j = x[1]
    return 10*i*np.random.random()-0.5 + 10*j*np.random.random()-0.5

dummy_subject_data = np.random.random((50,5,2))

#Test that should have at least some failures to do loglik function returning
#random values
def test_failure():
    with pytest.raises(ValueError):
        return mlepar(bad_loglik, dummy_subject_data, nparams=2,maxstarts=2)