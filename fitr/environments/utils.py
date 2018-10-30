# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
import numpy as np
from fitr.data import BehaviouralData
from fitr.data import merge_behavioural_data

def generate_behavioural_data(environment, agent, nsubjects, ntrials):
    """
    A function for flexibly simulating data for different task/agent combos.

    Arguments:

        environment: `fitr.environments.Graph` object
        agent: A `fitr.agents.Agent` object representing the agent being evaluated
        nsubjects: An `int` number of subjects to simulate
        ntrials: An `int` number of trials to simulate

    Returns:

        A `BehaviouralData` object containing all data simulated from the current run.

    Examples:

        ```python
        from fitr.agents import RWSoftmaxAgent
        from fitr.environments import TwoArmedBandit

        data = generate_behavioural_data(TwoArmedBandit, RWSoftmaxAgent, 5, 100)
        ```
    """
    for i in range(nsubjects):
        agent_ = agent(environment())
        subject_data = agent_.generate_data(ntrials)
        if i == 0:
            data = subject_data
        else:
            data = merge_behavioural_data([data, subject_data])
    return data

def reward_reflection(x, lb, ub):
    """ Imposes reflective boundaries on drifting reward functions.

    Denoting the lower bound by $l$ and the upper bound by $u$, this is computed according to the following formula:

    $$
    \max \Big\{\min \big\{\mathbf x, \max \big\{2u-\mathbf x, l \big\} \big\}, \min \big\{2l-\mathbf x, u \big\} \Big\}
    $$

    Arguments:

        x: An `ndarray((n,))` vector of values
        lb: A `float` depicting the lower bound for the rewards
        ub: A `float` depicting the upper bound for rewards

    Return:

        An updated reward vector `ndarray((n,))`
    """
    q = np.minimum(2*lb-x, ub)
    h = np.maximum(2*ub-x, lb)
    g = np.minimum(x, h)
    return np.maximum(g, q)
