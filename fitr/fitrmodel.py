class fitrmodel:
    """
    Object representing a reinforcement learning model for behavioural data of a specified task.

    Attributes
    ----------
    label : str
        String defining a name for the model.


    Methods
    -------
    fit(data, inference)
        Compute parameter estimates for the model.
    """
    def __init__(self, label=None):

        self.label = label


    def fit(self, data, method='eml', maxiter_eml=100):
        """
        Estimates parameter values of a reinforcement learning model given behavioural data from subjects

        Parameters
        ----------
        data : dict
            Subject level data. Specific fields are subject to the requirements of the task being modeled.
        method : {'eml', 'empirical'}
            Type of inference method to be employed:
                - 'eml': Expectation-maximization with Laplace approximation
                - 'empirical': Empirical priors
        maxiter_eml : int
            Maximum number of iterations (only valid if `inference='eml'`).

        Returns
        -------
        results : fitrfit
            Object containing the optimization output
        """

        # Initialize results object
        results = fitrfit()

        nsubjects = len(data)

        if method == 'eml':
            #  Set up the opimization protocol for the EML method. Includes
            #   A) Set maximum number of iterations of hyperparam estimates
            #   B) Initializing hyperparameters
            #   C) Initializing the eml log posterior function
            maxiter = maxiter_eml

        else:
            maxiter = 1

        while opt_iter < maxiter:
            for i in range(nsubjects):
                print('MODEL: ' + self.label + ', ITERATION ' + str(opt_iter) + ': Fitting Subject ' + str(i+1))

                _logpost = lambda x: -loglikelihood(states=data[i]['S'], actions=data[i]['A'], params=x) - prior()

            opt_iter += 1



class fitrfit:
    """
    Class representing the results of a fitrmodel optimization.

    Attributes
    ----------
    method : str
        Method employed in optimization.
    nsubjects : int
        Number of subjects fitted.
    nparams : int
        Number of free parameters in the fitted model.
    params : ndarray(shape=(nsubjects, nparams))
        Array of parameter estimates
    errs : ndarray(shape=(nsubjects, nparams))
        Array of parameter estimate errors
    LL : float
        Model log-likelihood
    LME : float
        Log-model evidence
    BIC : ndarray(shape=(nsubjects))
        Subject-level Bayesian Information Criterion
    AIC : ndarray(shape=(nsubjects))
        Subject-level Aikake Information Criterion

    Methods
    -------
    """
    def __init__(self, method, nsubjects, nparams):

        self.method = method
        self.nsubjects = nsubjects
        self.nparams = nparams
        self.params = np.zeros([nsubjects, nparams])
        self.errs = np.zeros([nsubjects, nparams])
        self.LL = None
        self.LME = None
        self.BIC = np.zeros(nsubjects)
        self.AIC = np.zeros(nsubjects)
