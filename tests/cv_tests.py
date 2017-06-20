# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from fitr.models import driftbandit as db
from fitr.model_selection import LOACV

def test_driftbandit_loacv():
    N = 5 # number of subjects
    T = 20 # number of trials
    narms = 2

    task = db.lr_cr_p(narms=2)
    res = task.simulate(nsubjects=N, ntrials=T)
    cv = LOACV(cv_func=task.loacv)
    cv.run(params=res.params, data=res.data)

    cv.results.accuracy_maplot(save_figure=True)
    plt.close()

    cv.results.accuracy_hist(save_figure=True)
    plt.close()

    cv.results.accuracy_param_scatter(save_figure=True, ylim=(0, 1))
    plt.close()
