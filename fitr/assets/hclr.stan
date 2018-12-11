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
