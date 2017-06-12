import numpy as np

from fitr.models import twostep
from fitr.metrics import parameter_distance
from fitr.unsupervised import Embedding
from fitr.unsupervised import TSNE
from fitr.unsupervised import AffinityPropagation
from fitr.unsupervised import Cluster

def test_embedding():
    emb = Embedding()
    assert(emb.algorithm is None)
    assert(emb.embedding is None)

def test_tsne():
    nsubjects = 20
    ntrials = 10

    res = twostep.lr_cr_mf().simulate(nsubjects=nsubjects, ntrials=ntrials)
    D = parameter_distance(params=res.params)
    group_labels = np.zeros(nsubjects)
    group_labels[10:] = 1

    tsne = TSNE()
    tsne.embed(D)

    assert(np.shape(tsne.embedding) == (nsubjects, 2))

    tsne.plot(group_labels=group_labels, show_figure=False)
    tsne.plot(show_figure=False)

def test_affinity_propagation():
    nsubjects = 20
    ntrials = 10

    res = twostep.lr_cr_mf().simulate(nsubjects=nsubjects, ntrials=ntrials)
    D = parameter_distance(params=res.params)
    group_labels = np.zeros(nsubjects)
    group_labels[10:] = 1

    ap = AffinityPropagation()
    ap.fit(data=D)
    ap.performance(group_labels=group_labels)
    ap.results

    ap2 = AffinityPropagation()
    ap2.fit(data=D)
    ap2.performance()
    ap2.results

def test_cluster():
    c = Cluster()
    assert(c.algorithm is None)
    assert(c.clusters is None)
    assert(c.results is None)
