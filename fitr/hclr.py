import pystan
import numpy as np

# ==============================================================================
#   STAN CODE FOR THE MODEL
# ==============================================================================

stancode = """
data {
  int<lower = 1> n_s;                   // n subjects
  int<lower = 1> n_t;                   // n trials
  int<lower = 1> n_f;                   // n parameters (+ 1 for intercept)
  int<lower = 1> n_c;                   // n covariates
  int<lower = 1> n_v;                   // n basis vectors
  vector[n_c] Z[n_s];                   // covariates
  matrix[n_t, n_f] X[n_s];              // convolution features
  int<lower=0, upper=1> y[n_s, n_t];    // actions taken
  row_vector[n_f] V[n_v];               // basis vectors
  real<lower=0> K_scale; 		            // width of prior on standardized effects
}
parameters {
  // Group level
  row_vector[n_f] mu;             // mean parameters (+ intercept)
  row_vector<lower=0>[n_f] sigma; // parameter sd (+ intercept)
  matrix[n_f, n_c] K;             // covariate weights
  row_vector[n_f] W_raw[n_s];     // unshifted individual parameters

}
transformed parameters {
  row_vector[n_f] W[n_s]; // shifted parameters

  // shift individual parameters according to group
  for (i in 1:n_s) {
    W[i] = mu + (K*Z[i])' + sigma .* W_raw[i];
  }

}

model {
  // Priors
  mu ~ cauchy(0, 5);
  sigma ~ cauchy(0, 5);

  for (i in 1:n_f) {
  	for (j in 1:n_c) {
  		K[i, j] ~ normal(0, K_scale);
  	}
  }

  for (i in 1:n_s) {
	W_raw[i] ~ normal(0, 1);
  }

  // Likelihood
  for (i in 1:n_s) {
    y[i] ~ bernoulli_logit(X[i]*W[i]');
  }

}

generated quantities {
  vector[n_v] group_indices;              // projections of group-level mean onto the basis
  row_vector[n_c] covariate_effects[n_v]; // projections of loading matrix onto the basis


  // Compute index values for overall group
  for (i in 1:n_v) {
    group_indices[i] = dot_product(mu, V[i]');
    covariate_effects[i] = V[i]*K;
  }
}

"""

# ==============================================================================
#   The main HCLR object
# ==============================================================================

class HCLR(object):
    """ Hierarchical Convolutional Logistic Regression (HCLR) for general behavioural data.


    Attributes:

        X: `ndarray((nsubjects,ntrials,nfeatures))`. The ``experience'' tensor.
        y: `ndarray((nsubjects,ntrials,ntargets))`. Tensor of ``choices'' we are trying to predict.
        Z: `ndarray((nsubjects,ncovariates))`. Covariates of interest
        V: `ndarray((naxes,nfeatures))`. Vectors identifying features of interest (i.e. to compute indices). If `add_intercept=True`, then the dimensionality of `V` should be `ndarray((naxes, nfeatures+1))`, where the first column represents the basis coordinate for the bias.
        filter_size: `int`. Number of steps prior to target included as features.
        loading_matrix_scale: `float > 0`. Scale of the loading matrix $\\boldsymbol\\Phi$, which is assumed that $\\phi_{ij} \\sim \\mathcal N(0, 1)$, with the default scale being 1.
        add_intercept: `bool'. Whether to add intercept
        group_mean: `ndarray`. Samples of the posterior group-level mean. `None` until model is fit
        group_scale: `ndarray`. Samples of the posterior group-level scale. `None` until model is fit
        loading_matrix: `ndarray`. Samples of the posterior loading matrix. `None` until model is fit
        subject_parameters: `ndarray`. Samples of the posterior subject-level parameters. `None` until model is fit
        group_indices: `ndarray`. Samples of the posterior group-level projections on to the basis. `None` until model is fit
        covariate_effects: `ndarray`. Samples of the posterior projection of the loading matrix onto the basis. `None` until model is fit

    ## Notes

    - When presenting `X` and `y`, note that the indices of `y` should correspond exactly to the trial indices in `X`, even though the HCLR analysis is predicting a trial ahead. In other words, there should be no lag in the `X`, `y` inputs. The HCLR setup will automatically set up the lag depending on how you set the `filter_size`.


    """
    def __init__(self,
                 X,
                 y,
                 Z,
                 V,
                 filter_size=5,
                 loading_matrix_scale=1.0,
                 add_intercept=True):
        self.y = y
        self.Z = Z
        self.filter_size = filter_size
        self.loading_matrix_scale = loading_matrix_scale
        self.add_intercept = add_intercept

        self.X = self._flatten_feature_representation(X, filter_size, add_intercept)

        # Prune the X and y data so that X[trial=t] lines up with y[trial=t+1]
        self.X = self.X[:,:-1,:]
        self.y = self.y[:,filter_size+1:]

        # Compute dimensions
        self.nsubjects, self.ntrials, self.nfeatures = self.X.shape
        self.ncovariates = self.Z.shape[1]

        # Set up the basis vectors
        if add_intercept:
            self.V = V[:,1:]
            self.v_bias = V[:,0].reshape(-1, 1)
        else:
            self.V = V
            self.v_bias = None

        self.V = self._expand_basis_vectors(filter_size)
        self.naxes = self.V.shape[0]

        self.data = self._make_data_dict()
        self.stancode = stancode

        # Initialize the posterior sample objects
        self.group_mean = None
        self.group_scale = None
        self.loading_matrix = None
        self.subject_parameters = None
        self.group_indices = None
        self.covariate_effects = None


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
            'V'  : self.V,
            'K_scale': self.loading_matrix_scale}
        return data


    def fit(self,
            nchains=4,
            niter=1000,
            warmup=None,
            thin=1,
            seed=None,
            verbose=False,
            algorithm='NUTS',
            n_jobs=-1):
        """ Fits the HCLR model

        Arguments:

            nchains: `int`. Number of chains for the MCMC run.
            niter: `int`. Number of iterations over which to run MCMC.
            warmup: `int`. Number of warmup iterations
            thin:  `int`. Periodicity of sample recording
            seed: `int`. Seed for pseudorandom number generator
            algorithm: `{'NUTS','HMC'}`
            n_jobs: `int`. Number of cores to use (default=-1, as many as possible and required)


        """
        self.model = pystan.StanModel(model_code=self.stancode)
        self.stanres = self.model.sampling(self.data,
                                           chains=nchains,
                                           iter=niter,
                                           warmup=warmup,
                                           thin=thin,
                                           seed=seed,
                                           verbose=verbose,
                                           algorithm=algorithm,
                                           n_jobs=n_jobs)

        # Extract results into attributes of the object
        self.group_mean = self.stanres.extract(['mu'])['mu']
        self.group_scale = self.stanres.extract(['sigma'])['sigma']
        self.loading_matrix = self.stanres.extract(['K'])['K']
        self.subject_parameters = self.stanres.extract(['W'])['W']
        self.group_indices = self.stanres.extract(['group_indices'])['group_indices']
        self.covariate_effects = self.stanres.extract(['covariate_effects'])['covariate_effects']
