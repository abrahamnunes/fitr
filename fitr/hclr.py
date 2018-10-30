import pystan 
import numpy as np 

class HCLR(object):
    """ Hierarchical Convolutional Logistic Regression (HCLR) for general behavioural data.
    """
    def __init__(self,
                 X, 
                 y, 
                 Z,
                 V,
                 filter_size=5, 
                 add_intercept=True):
        """
        
        Arguments: 

            X: `ndarray((nsubjects,ntrials,nfeatures))`. The ``experience'' tensor. 
            y: `ndarray((nsubjects,ntrials,ntargets))`. Tensor of ``choices'' we are trying to predict. 
            Z: `ndarray((nsubjects,ncovariates))`. Covariates of interest 
            V: `ndarray((naxes,nfeatures))`. Vectors identifying features of interest (i.e. to compute ``indices.'' If `add_intercept=True`, then the dimensionality of `V` should be `ndarray((naxes, nfeatures+1))`, where the first column represents the basis coordinate for the bias.  
            filter_size: `int`. Number of steps prior to target included as features.
            add_intercept: `bool'. Whether to add intercept

        """

        self.y = y
        self.Z = Z
        self.filter_size = filter_size 
        self.add_intercept = add_intercept
        
        self.X = self._flatten_feature_representation(X, filter_size, add_intercept)
        self.nsubjects, self.ntrials, self.nfeatures = self.X.shape
        self.ncovariates = self.Z.shape[1]
        
        if add_intercept: 
            self.V = V[:,1:]
            self.v_bias = V[:,0].reshape(-1, 1)
        else: 
            self.V = V
            self.v_bias = None
            
        self.V = self._expand_basis_vectors(filter_size)
        self.naxes = self.V.shape[0]
        
        self.data = self._make_data_dict()
        self.stancode = 'INSERT CODE HERE'

    def _expand_basis_vectors(self, filter_size):
        """ Tiles the basis (feature template) vectors to make them of appropriate dimension 

        Arguments: 
            
            filter_size: `int`. Number of steps prior to target included as features.

        Returns: 

            V: `ndarray((nfeatures*filter_size+/-1,naxes))`. Vectors identifying features of interest (i.e. to compute ``indices,'' now sized appropriately for the flattened version of `X`. Note that here `+/-1` in the `ndarray` dimension refers to the potentially increased number of columns if a bias term is included in `X`.

        """
        V = None
        for i in range(filter_size):
            if V is None: 
                V = self.V
            else: 
                V = np.hstack((V, self.V))

        if self.v_bias is not None: 
            V = np.hstack((self.v_bias, V))

        return V

    def _flatten_feature_representation(self, X, filter_size, add_intercept): 
        """ Facilitates formulation of convolution as matrix multiplication 
        
        Arguments: 


            X: `ndarray((nsubjects,ntrials,nfeatures))`. The ``experience'' tensor. 
            filter_size: `int`. Number of steps prior to target included as features.
            add_intercept: `bool`. Whether intercept will be added 

        Returns: 

            X: `list` of size `nsubjects` containing `ndarray((ntrials-filter_size, nfeatures*filter_size))`. Matrix-representation of data for convolutional analysis
        """ 
        nsubjects = len(X)
        X_ = []
        for i in range(nsubjects): 
            ntrials, nfeatures = X[i].shape
            Xi = []
            for j in range(filter_size, ntrials):
                Xi.append(np.ravel(X[i][j-filter_size:j,:]))
            
            Xi = np.stack(Xi)
            
            if add_intercept: 
                Xi = np.hstack((np.ones((Xi.shape[0], 1)), Xi))
            
            X_.append(Xi)
            
        return np.array(X_)

    def _make_data_dict(self): 
        """ Creates dictionary format data for Stan """ 

        data = {
            'n_s': self.nsubjects, 
            'n_t': self.ntrials, 
            'n_f': self.nfeatures, 
            'n_c': self.ncovariates,
            'n_v': self.naxes, 
            'X'  : self.X, 
            'y'  : self.y,
            'Z'  : self.Z, 
            'V'  : self.V}
        return data 


    def fit(self, nchains=4, niter=1000, algorithm='NUTS'):
        """ Fits the HCLR model
        
        Arguments: 

            nchains: `int`. Number of chains for the MCMC run. 
            niter: `int`. Number of iterations over which to run MCMC. 

        """
        self.model = pystan.StanModel(model_code=self.stancode)
        self.stanres = self.model.sampling()
