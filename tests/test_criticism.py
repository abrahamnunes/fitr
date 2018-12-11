import numpy as np
import matplotlib.pyplot as plt
from fitr.criticism.plotting import make_diagline_range
from fitr.criticism.plotting import actual_estimate

def test_actual_estimate():
    x = np.linspace(0, 10, 20)
    f = actual_estimate(x, x, xlabel='x', ylabel='y')
    del(f)
    plt.close()
