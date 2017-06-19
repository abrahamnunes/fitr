# -*- coding: utf-8 -*-
import numpy as np

from fitr.inference import MCMC
from fitr.models import driftbandit as db

def test_mcmc():
	nsubjects = 5
	ntrials = 10

	taskresults = db.lr_cr(narms=4).simulate(nsubjects=nsubjects, ntrials=ntrials)

	banditgm = db.lr_cr(narms=4).gm
	model = MCMC(generative_model=banditgm)

	assert(model.name == 'FitrMCMCModel')
	assert(model.generative_model == banditgm)

	lrcr = model.fit(data=taskresults.data_mcmc, chains=1)
	lrcr.get_paramestimates()
	lrcr.trace_plot(save_figure=True)

	assert(lrcr.name == 'FitrMCMCModel')
	assert(lrcr.method == 'MCMC')
	assert(lrcr.nsubjects == 5)
	assert(lrcr.nparams == 2)
	assert(np.shape(lrcr.params) == (5, 2))
	assert(len(lrcr.paramnames) == 2)
