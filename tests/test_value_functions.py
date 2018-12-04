import numpy as np 
from fitr.environments import TwoArmedBandit 
from fitr.agents.value_functions import AsymmetricRescorlaWagnerLearner
from fitr.agents.value_functions import ForgetfulInstrumentalRescorlaWagnerLearner
from fitr.agents.value_functions import ForgetfulAsymmetricRescorlaWagnerLearner

def test_asymmetric_rwlearner(): 
    task = TwoArmedBandit()
    critic = AsymmetricRescorlaWagnerLearner(task, learning_rate_pos=0.1, learning_rate_neg=0.1)
    x = task.observation()
    u = task.random_action()
    x_, r, _ = task.step(u)
    critic.update(x, u, r, x_, None)


def test_forgetful_asymmetric_rwlearner(): 
    task = TwoArmedBandit()
    critic = ForgetfulAsymmetricRescorlaWagnerLearner(task, learning_rate_pos=0.1, learning_rate_neg=0.1, memory_decay=0.9)
    x = task.observation()
    u = task.random_action()
    x_, r, _ = task.step(u)
    critic.update(x, u, r, x_, None)


def test_forgetful_rwlearner(): 
    task = TwoArmedBandit()
    critic = ForgetfulInstrumentalRescorlaWagnerLearner(task, learning_rate=0.1, memory_decay=0.1)
    x = task.observation()
    u = task.random_action()
    x_, r, _ = task.step(u)
    critic.update(x, u, r, x_, None)
