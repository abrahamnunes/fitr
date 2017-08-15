import pytest
import numpy as np
import fitr
from fitr.rlparams import *
from fitr.models import twostep as task
from fitr.criticism.distance import parameter_distance
from fitr.criticism.distance import likelihood_distance
from fitr.plotting import heatmap
from fitr.plotting import confusion_matrix
from fitr.plotting import distance_scatter
from fitr.plotting import distance_hist

def test_param_plot_pdf():
    LearningRate(mean=0.5, sd=0.2).plot_pdf()
    ChoiceRandomness(mean=4.5, sd=2).plot_pdf()
    Perseveration().plot_pdf()

    # Test the exceptions
    with pytest.raises(Exception):
        LearningRate().plot_pdf(xlim=[1, 0], save_figure=True)

    with pytest.raises(Exception):
        LearningRate().plot_pdf(xlim=[-1, 1], save_figure=True)

    with pytest.raises(Exception):
        LearningRate().plot_pdf(xlim=[0, 2], save_figure=True)

    with pytest.raises(Exception):
        ChoiceRandomness().plot_pdf(xlim=[-1, 20], save_figure=True)

    with pytest.raises(Exception):
        ChoiceRandomness().plot_pdf(xlim=[1, -20], save_figure=True)

def test_synthetic_data_plots():
    group = task.lr_cr_mf().simulate(ntrials=20, nsubjects=5)
    f = group.plot_cumreward(save_figure=True)
    f = group.cumreward_param_plot(save_figure=True)

def test_distance_plots(tmpdir):
    nsubjects = 20
    res = task.lr_cr_mf().simulate(ntrials=20, nsubjects=nsubjects)
    param_dist = parameter_distance(params=res.params)
    ll_dist = likelihood_distance(loglik_func=task.lr_cr_mf().loglikelihood,
                                  params=res.params,
                                  data=res.data)

    group_labels = np.zeros(nsubjects)
    group_labels[10:] = 1

    _file = tmpdir.join('output.pdf')
    heatmap(param_dist,
            title='Heatmap',
            xlab='X',
            ylab='Y',
            interpolation='none',
            save_figure=True,
            figname=_file.strpath)

    _file = tmpdir.join('output.pdf')
    distance_scatter(param_dist,
                     ll_dist,
                     group_labels=group_labels,
                     alpha=0.5,
                     save_figure=True,
                     figname=_file.strpath)

    _file = tmpdir.join('output.pdf')
    distance_hist(param_dist,
                  group_labels=group_labels,
                  alpha=0.5,
                  save_figure=True,
                  figname=_file.strpath)

def test_confusion_matrix():
    X = np.random.uniform(0, 1, size=(10, 10))
    fig = confusion_matrix(X=X,
                           classes=np.arange(10),
                           normalize=False,
                           round_digits=2,
                           title='Confusion matrix',
                           xlabel='Predicted Label',
                           ylabel='True Label',
                           file_dir='tmp/',
                           filename='figsaved.pdf',
                           cmap=plt.cm.Blues)
    fig = confusion_matrix(X=X,
                           classes=np.arange(10),
                           normalize=True,
                           round_digits=2,
                           title='Confusion matrix',
                           xlabel='Predicted Label',
                           ylabel='True Label',
                           file_dir='tmp/',
                           filename='figsaved.pdf',
                           cmap=plt.cm.Blues)
