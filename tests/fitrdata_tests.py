# -*- coding: utf-8 -*-

import os
import pytest
import numpy as np

from fitr.rlparams import *
from fitr.models import TaskModel
from fitr.models import combine_groups
from fitr.models import twostep as task

def test_taskmodel():
    task = TaskModel()

    with pytest.raises(Exception):
        task.set_gm(path=os.path.join("mycode", "datacode.txt"),
                    paramnames_long=['A', 'B'],
                    paramnames_code=['a', 'b'])


def test_synthdata_and_combine():
    ntrials = 10
    nsubjects = 5
    group1_task = task.lr_cr_mf(LR=LearningRate(mean=0.3, sd=0.05),
                                CR=ChoiceRandomness(mean=3, sd=1))
    group1_data = group1_task.simulate(ntrials=ntrials,
                                       nsubjects=nsubjects,
                                       group_id='HC')
    group2_task = task.lr_cr_mf(LR=LearningRate(mean=0.1, sd=0.05),
                                CR=ChoiceRandomness(mean=8, sd=2))
    group2_data = group2_task.simulate(ntrials=ntrials,
                                       nsubjects=nsubjects,
                                       group_id='EDO')

    pooled_data = combine_groups(x=group1_data, y=group2_data)

    assert(len(pooled_data.data) == nsubjects*2)
    assert(pooled_data.data_mcmc['T'] == ntrials)
    assert(pooled_data.data_mcmc['N'] == nsubjects*2)
    assert(np.shape(pooled_data.data_mcmc['S2']) == (ntrials, nsubjects*2))
    assert(np.shape(pooled_data.data_mcmc['A1']) == (ntrials, nsubjects*2))
    assert(np.shape(pooled_data.data_mcmc['A2']) == (ntrials, nsubjects*2))
    assert(np.shape(pooled_data.data_mcmc['R']) == (ntrials, nsubjects*2))
    assert(np.size(pooled_data.data_mcmc['G']) == nsubjects*2)

    # Test the get_nparams and get_nsubjects methods
    n_params = pooled_data.get_nparams()
    n_subj   = pooled_data.get_nsubjects()

    assert(n_params == 2)
    assert(n_subj == nsubjects*2)
