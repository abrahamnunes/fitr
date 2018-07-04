import time 
import numpy as np
import matplotlib.pyplot as plt 

from fitr.utils import logsumexp as fitr_logsumexp
from scipy.special import logsumexp as scipy_logsumexp

n = 10000
X = np.empty((n, 2))
for i in range(n):
    x = np.ones(i+2)
    ft0 = time.time()
    yf  = fitr_logsumexp(x)
    ft  = time.time()-ft0

    fs0 = time.time()
    ys  = scipy_logsumexp(x)
    fs  = time.time()-fs0

    X[i]= np.array([fs, ft])


fig, ax = plt.subplots(figsize=(7, 3))
ax.set_title('Log-sum-exp Speed')
ax.set_xlabel(r'Dimensionality of $\mathbf{x}$')
ax.set_ylabel(r'Time (s)')
ax.semilogy(np.arange(n)+2, X[:,1], c='k', ls='-', label='fitr')
ax.semilogy(np.arange(n)+2, X[:,0], c='k', ls='--', label='SciPy')
plt.show()
