import pytest
import fitr
from fitr.rlparams import *
import fitr.twostep as task

def test_param_plot_pdf():
    LearningRate(mean=0.5, sd=0.2).plot_pdf()
    ChoiceRandomness(mean=4.5, sd=2).plot_pdf()

    with pytest.raises(Exception):
        LearningRate().plot_pdf(xlim=[1, 0])
        LearningRate().plot_pdf(xlim=[-1, 1])
        LearningRate().plot_pdf(xlim=[0, 2])
        ChoiceRandomness().plot_pdf(xlim=[-1, 20])
        ChoiceRandomness().plot_pdf(xlim=[1, -20])

def test_synthetic_data_plots():
    group = task.lr_cr_mf().simulate(ntrials=20, nsubjects=5)

    group.plot_cumreward()
    group.cumreward_param_plot()
