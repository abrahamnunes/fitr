"""
Objects representing each parameter object
"""
import scipy.stats

class Param(object):
    def __init__(self, name=None, rng=None):
        self.name  = name
        self.range = rng
        self.dist  = None

class LearningRate(Param):
    """
    A learning rate object.
    """
    def __init__(self, name='Learning Rate', rng='unit'):
        self.name = name
        self.rng  = rng
        self.dist = scipy.stats.beta(1.1, 1.1)

class RewardSensitivity(Param):
    """
    A reward sensitivity object.
    """
    def __init__(self, name='Reward Sensitivity', rng='unit'):
        self.name = name
        self.rng  = rng
        self.dist = scipy.stats.beta(1.1, 1.1)

class EligibilityTrace(Param):
    """
    An eligibility trace parameter object.
    """
    def __init__(self, name='Eligibility Trace', rng='unit'):
        self.name = name
        self.rng  = rng
        self.dist = scipy.stats.beta(1.1, 1.1)

class MBMF_Balance(Param):
    """
    An object representing the parameter that balances model-based and model-free control.
    """
    def __init__(self, name='Model-Based Control Weight', rng='unit'):
        self.name = name
        self.rng  = rng
        self.dist = scipy.stats.beta(1.1, 1.1)

class ChoiceRandomness(Param):
    """
    An choice randomness parameter object
    """
    def __init__(self, name='Choice Randomness', rng='pos'):
        self.name = name
        self.rng  = rng
        self.dist = scipy.stats.gamma(5, 1)
