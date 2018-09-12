import numpy as np
from fitr.environments import TwoStep


def test_set_seed():
    task = TwoStep()
    task.set_seed(235)

    task2 = TwoStep()
    task2.set_seed(235)

    state1 = task.rng.get_state()
    state2 = task2.rng.get_state()
    assert(np.all(state1[1] == state2[1]))
