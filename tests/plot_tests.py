import pytest
import numpy as np
import fitr
from fitr.rlparams import *
from fitr.models import twostep as task
from fitr.metrics import parameter_distance
from fitr.metrics import likelihood_distance
from fitr.plotting import heatmap
from fitr.plotting import distance_scatter
from fitr.plotting import distance_hist

def test_param_plot_pdf():
    LearningRate(mean=0.5, sd=0.2).plot_pdf(show_figure=False)
    ChoiceRandomness(mean=4.5, sd=2).plot_pdf(show_figure=False)
    Perseveration().plot_pdf(show_figure=False)

    with pytest.raises(Exception):
        LearningRate().plot_pdf(xlim=[1, 0], show_figure=False)
        LearningRate().plot_pdf(xlim=[-1, 1], show_figure=False)
        LearningRate().plot_pdf(xlim=[0, 2], show_figure=False)
        ChoiceRandomness().plot_pdf(xlim=[-1, 20], show_figure=False)
        ChoiceRandomness().plot_pdf(xlim=[1, -20], show_figure=False)

def test_synthetic_data_plots():
    group = task.lr_cr_mf().simulate(ntrials=20, nsubjects=5)

    group.plot_cumreward(show_figure=False)
    group.cumreward_param_plot(show_figure=False)

def test_distance_plots():
    nsubjects = 20
    res = task.lr_cr_mf().simulate(ntrials=20, nsubjects=nsubjects)
    param_dist = parameter_distance(params=res.params)
    ll_dist = likelihood_distance(loglik_func=task.lr_cr_mf().loglikelihood,
                                  params=res.params,
                                  data=res.data)

    group_labels = np.zeros(nsubjects)
    group_labels[10:] = 1
    heatmap(param_dist,
            title='Heatmap',
            xlab='X',
            ylab='Y',
            show_figure=False)
    distance_scatter(param_dist,
                     ll_dist,
                     group_labels=group_labels,
                     show_figure=False)
    distance_hist(param_dist, group_labels=group_labels, show_figure=False)
