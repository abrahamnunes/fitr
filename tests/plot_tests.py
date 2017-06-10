import pytest
import fitr
from fitr.rlparams import *
import fitr.twostep as task

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
