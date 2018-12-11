import numpy as np
from fitr.environments import DawTwoStep
from fitr.environments import IGT
from fitr.environments import MouthTask
from fitr.environments import OrthogonalGoNoGo
from fitr.environments import TwoArmedBandit
from fitr.environments import RandomContextualBandit


def test_set_seed():
    task = DawTwoStep()
    task.set_seed(235)

    task2 = DawTwoStep()
    task2.set_seed(235)

    state1 = task.rng.get_state()
    state2 = task2.rng.get_state()
    assert(np.all(state1[1] == state2[1]))

    # Get graph depth
    d = task.get_graph_depth()

    # Test figures
    f = task.plot_graph()
    del(f)

    f = task.plot_spectral_properties()
    del(f)

    f = task.plot_action_outcome_probabilities(outfile=None)
    del(f)

def test_igt():
    task = IGT()
    x = task.observation()
    u = task.random_action()

def test_mouthtask():
    task = MouthTask()
    x = task.observation()
    u = task.random_action()

def test_two_armed_bandit():
    task = TwoArmedBandit()
    x = task.observation()
    u = task.random_action()

def test_ogng():
    task = OrthogonalGoNoGo()
    x = task.observation()
    u = task.random_action()

def test_random_contextual_bandit():
    task = RandomContextualBandit(3, 5, 3, 3)
    x = task.observation()
    u = task.random_action()
