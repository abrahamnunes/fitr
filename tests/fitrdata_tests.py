# -*- coding: utf-8 -*-

import fitr
from fitr.rlparams import *
import fitr.twostep as task
import numpy as np
import scipy

def test_syntheticdata():
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

    group1_data.append_group(data=group2_data)
    assert(len(group1_data.data) == nsubjects*2)
    assert(group1_data.data_mcmc['T'] == ntrials)
    assert(group1_data.data_mcmc['N'] == nsubjects*2)
    assert(np.shape(group1_data.data_mcmc['S2']) == (ntrials, nsubjects*2))
    assert(np.shape(group1_data.data_mcmc['A1']) == (ntrials, nsubjects*2))
    assert(np.shape(group1_data.data_mcmc['A2']) == (ntrials, nsubjects*2))
    assert(np.shape(group1_data.data_mcmc['R']) == (ntrials, nsubjects*2))
    assert(np.size(group1_data.data_mcmc['G']) == nsubjects*2)
