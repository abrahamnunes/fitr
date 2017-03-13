"""
Objects representing each parameter object
"""
import scipy.stats

class Param(object):
    """
    A base parameter object that can be used to generate new parameters.

    Attributes
    ----------
    name : str
        Name of the parameter. To be used for plots and so forth.
    rng : {'unit', 'pos', 'neg', 'unc'}
        The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
    """
    def __init__(self, name=None, rng=None):
        self.name  = name
        self.range = rng
        self.dist  = None

class LearningRate(Param):
    """
    A learning rate object.

    Attributes
    ----------
    name : str
        Name of the parameter. To be used for plots and so forth.
    rng : {'unit', 'pos', 'neg', 'unc'}
        The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
    dist : scipy.stats.beta distribution

    """
    def __init__(self, name='Learning Rate', rng='unit', shape_alpha=1.1, shape_beta=1.1):
        """
        Instantiates the Parameter

        Parameters
        ----------
        name : str
            Name of the parameter. To be used for plots and so forth.
        rng : {'unit', 'pos', 'neg', 'unc'}
            The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
        shape_alpha : float over domain [0, +Inf]
            The alpha parameter of the beta distribution
        shape_beta : float over domain [0, +Inf]
            The beta parameter of the beta distribution
        """
        self.name = name
        self.rng  = rng
        self.dist = scipy.stats.beta(shape_alpha, shape_beta)

class RewardSensitivity(Param):
    """
    A reward sensitivity object.

    Attributes
    ----------
    name : str
        Name of the parameter. To be used for plots and so forth.
    rng : {'unit', 'pos', 'neg', 'unc'}
        The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
    dist : scipy.stats.beta distribution

    """
    def __init__(self, name='Reward Sensitivity', rng='unit', shape_alpha=1.1, shape_beta=1.1):
        """
        Instantiates the Parameter

        Parameters
        ----------
        name : str
            Name of the parameter. To be used for plots and so forth.
        rng : {'unit', 'pos', 'neg', 'unc'}
            The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
        shape_alpha : float over domain [0, +Inf]
            The alpha parameter of the beta distribution
        shape_beta : float over domain [0, +Inf]
            The beta parameter of the beta distribution
        """
        self.name = name
        self.rng  = rng
        self.dist = scipy.stats.beta(shape_alpha, shape_beta)

class EligibilityTrace(Param):
    """
    An eligibility trace parameter object.

    Attributes
    ----------
    name : str
        Name of the parameter. To be used for plots and so forth.
    rng : {'unit', 'pos', 'neg', 'unc'}
        The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
    dist : scipy.stats.beta distribution

    """
    def __init__(self, name='Eligibility Trace', rng='unit', shape_alpha=1.1, shape_beta=1.1):
        """
        Instantiates the Parameter

        Parameters
        ----------
        name : str
            Name of the parameter. To be used for plots and so forth.
        rng : {'unit', 'pos', 'neg', 'unc'}
            The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
        shape_alpha : float over domain [0, +Inf]
            The alpha parameter of the beta distribution
        shape_beta : float over domain [0, +Inf]
            The beta parameter of the beta distribution
        """
        self.name = name
        self.rng  = rng
        self.dist = scipy.stats.beta(shape_alpha, shape_beta)

class MBMF_Balance(Param):
    """
    An object representing the parameter that balances model-based and model-free control.

    Attributes
    ----------
    name : str
        Name of the parameter. To be used for plots and so forth.
    rng : {'unit', 'pos', 'neg', 'unc'}
        The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
    dist : scipy.stats.beta distribution

    """
    def __init__(self, name='Model-Based Control Weight', rng='unit', shape_alpha=1.1, shape_beta=1.1):
        """
        Instantiates the Parameter

        Parameters
        ----------
        name : str
            Name of the parameter. To be used for plots and so forth.
        rng : {'unit', 'pos', 'neg', 'unc'}
            The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
        shape_alpha : float over domain [0, +Inf]
            The alpha parameter of the beta distribution
        shape_beta : float over domain [0, +Inf]
            The beta parameter of the beta distribution
        """

        self.name = name
        self.rng  = rng
        self.dist = scipy.stats.beta(shape_alpha, shape_beta)

class ChoiceRandomness(Param):
    """
    An choice randomness parameter object

    Attributes
    ----------
    name : str
        Name of the parameter. To be used for plots and so forth.
    rng : {'unit', 'pos', 'neg', 'unc'}
        The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
    dist : scipy.stats.gamma distribution

    """
    def __init__(self, name='Choice Randomness', rng='pos', shape=5., scale=1.):
        """
        Instantiates the Parameter

        Parameters
        ----------
        name : str
            Name of the parameter. To be used for plots and so forth.
        rng : {'unit', 'pos', 'neg', 'unc'}
            The domain over which the parameter lies (unit=[0,1], pos=[0,+Inf], neg=[-Inf,0], unc=[-Inf, +Inf])
        shape : float over domain [0, +Inf]
            The shape parameter of the gamma distribution
        scale : float over domain [0, +Inf]
            The scale parameter of the gamma distribution
        """
        self.name = name
        self.rng  = rng
        self.dist = scipy.stats.gamma(shape, scale)
