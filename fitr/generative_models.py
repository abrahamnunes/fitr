# ==============================================================================
#
#   GENERATIVE MODELS
#       Defines code for Stan models to be used by the MCMC method in fitr
#
# ==============================================================================

class GenerativeModel(object):
    """
    Base class for generative models

    Attributes
    ----------
    paramnames : dict
        Dictionary with two entries: 'long' which are strings representing parameter names, and 'code', which are strings denoting the parameter names as encoded in the Stan code
    model : string
        String representing Stan model code
    """
    def __init__(self):
        self.paramnames = {'long': [], 'code': []}
        self.model = ''

# ------------------------------------------------------------------------------
#
#   Two-step task
#
# ------------------------------------------------------------------------------

class twostep(GenerativeModel):
    def __init__(self, model='lr_cr_w'):

        if model == 'lr_cr_w':
            self.paramnames = {
                'long' : ['Learning Rate',
                          'Choice Randomness',
                          'Model-Based Weight'],
                'code' : ['lr', 'cr', 'w']
            }
            self.model = """
                data {
                    int<lower=1> N;
                    int<lower=1> T;
                    int<lower=1,upper=2> S2[T, N];
                    int<lower=1,upper=2> A1[T, N];
                    int<lower=1,upper=2> A2[T, N];
                    real R[T, N];
                }
                transformed data {
                    vector[2] Q_o[3];

                    for (i in 1:3){
                        Q_o[i] = rep_vector(0.0, 2);
                    }

                }
                parameters {
                    # Initialize group-level hyperparameters
                    vector[3] mu_p;
                    vector<lower=0>[3] sigma;

                    # Subject-level parameters
                    vector[N] lr_pr;
                    vector[N] cr_pr;
                    vector[N] w_pr;
                }
                transformed parameters {
                    # Subject-level parameters
                    vector<lower=0, upper=1>[N] lr;
                    vector<lower=0, upper=10>[N] cr;
                    vector<lower=0, upper=1>[N] w;

                    for (i in 1:N) {
                        lr[i] = Phi_approx(mu_p[1] + sigma[1]*lr_pr[i]);
                        cr[i] = Phi_approx(mu_p[2] + sigma[2]*cr_pr[i])*10;
                        w[i] = Phi_approx(mu_p[3] + sigma[3]*w_pr[i]);
                    }
                }
                model {
                    # Hyperparameters
                    mu_p ~ normal(0, 1);
                    sigma ~ cauchy(0, 5);

                    # Individual parameters
                    lr_pr ~ normal(0, 1);
                    cr_pr ~ normal(0, 1);
                    w_pr ~ normal(0, 1);

                    # Subject and trial loops
                    for (i in 1:N) {
                        vector[2] Q[3];
                        vector[2] Qmf[3];
                        vector[2] Qmb[3];
                        real PE;

                        Q = Q_o;
                        Qmf = Q_o;
                        Qmb = Q_o;

                        for (t in 1:T) {
                            # First state action
                            A1[t, i] ~ categorical_logit(cr[i]*Q[1]);

                            # Second state action
                            A2[t, i] ~ categorical_logit(cr[i]*Q[S2[t, i]]);

                            # Prediction error
                            PE = R[t, i] - Q[S2[t, i]][A2[t, i]];

                            # Learning (Model Free)
                            Qmf[1][A1[t, i]] = Qmf[1][A1[t, i]] + lr[i]*(Qmf[S2[t, i]][A2[t, i]] - Qmf[1][A1[t, i]]);
                            Qmf[S2[t, i]][A2[t, i]] = Qmf[S2[t, i]][A2[t, i]] + lr[i]*PE;
                            Qmf[1][A1[t, i]] = Qmf[1][A1[t, i]] + lr[i]*PE;

                            # Learning (Model Based)
                            Qmb[1][1] = 0.7*max(Qmf[2]) + 0.3*max(Qmf[3]);
                            Qmb[1][2] = 0.7*max(Qmf[3]) + 0.3*max(Qmf[2]);

                            # Mix MF and MB
                            Q[2:3] = Qmf[2:3];
                            Q[1] = w[i]*Qmb[1] + (1-w[i])*Qmf[1];
                        }
                    }
                }
            """

        elif model == 'lr_cr_et_w':
            self.paramnames = {
                'long' : ['Learning Rate',
                          'Choice Randomness',
                          'Eligibility Trace',
                          'Model-Based Weight'],
                'code' : ['lr', 'cr', 'et', 'w']
            }
            self.model = """
                data {
                    int<lower=1> N;
                    int<lower=1> T;
                    int<lower=1,upper=2> S2[T, N];
                    int<lower=1,upper=2> A1[T, N];
                    int<lower=1,upper=2> A2[T, N];
                    real R[T, N];
                }
                transformed data {
                    vector[2] Q_o[3];

                    for (i in 1:3){
                        Q_o[i] = rep_vector(0.0, 2);
                    }

                }
                parameters {
                    # Initialize group-level hyperparameters
                    vector[4] mu_p;
                    vector<lower=0>[4] sigma;

                    # Subject-level parameters
                    vector[N] lr_pr;
                    vector[N] cr_pr;
                    vector[N] et_pr;
                    vector[N] w_pr;
                }
                transformed parameters {
                    # Subject-level parameters
                    vector<lower=0, upper=1>[N] lr;
                    vector<lower=0, upper=10>[N] cr;
                    vector<lower=0, upper=1>[N] et;
                    vector<lower=0, upper=1>[N] w;

                    for (i in 1:N) {
                        lr[i] = Phi_approx(mu_p[1] + sigma[1]*lr_pr[i]);
                        cr[i] = Phi_approx(mu_p[2] + sigma[2]*cr_pr[i])*10;
                        et[i] = Phi_approx(mu_p[3] + sigma[3]*et_pr[i]);
                        w[i] = Phi_approx(mu_p[4] + sigma[4]*w_pr[i]);
                    }
                }
                model {
                    # Hyperparameters
                    mu_p ~ normal(0, 1);
                    sigma ~ cauchy(0, 5);

                    # Individual parameters
                    lr_pr ~ normal(0, 1);
                    cr_pr ~ normal(0, 1);
                    et_pr ~ normal(0, 1);
                    w_pr ~ normal(0, 1);

                    # Subject and trial loops
                    for (i in 1:N) {
                        vector[2] Q[3];
                        vector[2] Qmf[3];
                        vector[2] Qmb[3];
                        real PE;

                        Q = Q_o;
                        Qmf = Q_o;
                        Qmb = Q_o;

                        for (t in 1:T) {
                            # First state action
                            A1[t, i] ~ categorical_logit(cr[i]*Q[1]);

                            # Second state action
                            A2[t, i] ~ categorical_logit(cr[i]*Q[S2[t, i]]);

                            # Prediction error
                            PE = R[t, i] - Q[S2[t, i]][A2[t, i]];

                            # Learning (Model Free)
                            Qmf[1][A1[t, i]] = Qmf[1][A1[t, i]] + lr[i]*(Qmf[S2[t, i]][A2[t, i]] - Qmf[1][A1[t, i]]);
                            Qmf[S2[t, i]][A2[t, i]] = Qmf[S2[t, i]][A2[t, i]] + lr[i]*PE;
                            Qmf[1][A1[t, i]] = Qmf[1][A1[t, i]] + lr[i]*PE*et[i];

                            # Learning (Model Based)
                            Qmb[1][1] = 0.7*max(Qmf[2]) + 0.3*max(Qmf[3]);
                            Qmb[1][2] = 0.7*max(Qmf[3]) + 0.3*max(Qmf[2]);

                            # Mix MF and MB
                            Q[2:3] = Qmf[2:3];
                            Q[1] = w[i]*Qmb[1] + (1-w[i])*Qmf[1];
                        }
                    }
                }
            """

        elif model == 'lr_cr_w_p':
            self.paramnames = {
                'long' : ['Learning Rate',
                          'Choice Randomness',
                          'Model-Based Weight',
                          'Perseveration'],
                'code' : ['lr', 'cr', 'w', 'p']
            }
            self.model = """
                data {
                    int<lower=1> N;
                    int<lower=1> T;
                    int<lower=1,upper=2> S2[T, N];
                    int<lower=1,upper=2> A1[T, N];
                    int<lower=1,upper=2> A2[T, N];
                    real R[T, N];
                }
                transformed data {
                    vector[2] Q_o[3];

                    for (i in 1:3){
                        Q_o[i] = rep_vector(0.0, 2);
                    }

                }
                parameters {
                    # Initialize group-level hyperparameters
                    vector[4] mu_p;
                    vector<lower=0>[4] sigma;

                    # Subject-level parameters
                    vector[N] lr_pr;
                    vector[N] cr_pr;
                    vector[N] w_pr;
                    vector[N] persev_pr;
                }
                transformed parameters {
                    # Subject-level parameters
                    vector<lower=0, upper=1>[N] lr;
                    vector<lower=0, upper=10>[N] cr;
                    vector<lower=0, upper=1>[N] w;
                    vector[N] persev;

                    for (i in 1:N) {
                        lr[i] = Phi_approx(mu_p[1] + sigma[1]*lr_pr[i]);
                        cr[i] = Phi_approx(mu_p[2] + sigma[2]*cr_pr[i])*10;
                        w[i] = Phi_approx(mu_p[3] + sigma[3]*w_pr[i]);
                        persev[i] = mu_p[4] + sigma[4]*persev_pr[i];
                    }
                }
                model {
                    # Hyperparameters
                    mu_p ~ normal(0, 1);
                    sigma ~ cauchy(0, 5);

                    # Individual parameters
                    lr_pr ~ normal(0, 1);
                    cr_pr ~ normal(0, 1);
                    w_pr ~ normal(0, 1);
                    persev_pr ~ normal(0, 1);

                    # Subject and trial loops
                    for (i in 1:N) {
                        vector[2] Q[3];
                        vector[2] Qmf[3];
                        vector[2] Qmb[3];
                        real PE;
                        int a_last;
                        real repr;

                        Q = Q_o;
                        Qmf = Q_o;
                        Qmb = Q_o;

                        a_last = 100;
                        for (t in 1:T) {
                            # Compute perseveration function
                            if (A1[t, i] == a_last)
                              repr = 1;
                            else
                              repr = 0;

                            # First state action
                            A1[t, i] ~ categorical_logit(cr[i]*Q[1] + persev[i]*repr);

                            # Second state action
                            A2[t, i] ~ categorical_logit(cr[i]*Q[S2[t, i]]);

                            # Prediction error
                            PE = R[t, i] - Q[S2[t, i]][A2[t, i]];

                            # Learning (Model Free)
                            Qmf[1][A1[t, i]] = Qmf[1][A1[t, i]] + lr[i]*(Qmf[S2[t, i]][A2[t, i]] - Qmf[1][A1[t, i]]);
                            Qmf[S2[t, i]][A2[t, i]] = Qmf[S2[t, i]][A2[t, i]] + lr[i]*PE;
                            Qmf[1][A1[t, i]] = Qmf[1][A1[t, i]] + lr[i]*PE;

                            # Learning (Model Based)
                            Qmb[1][1] = 0.7*max(Qmf[2]) + 0.3*max(Qmf[3]);
                            Qmb[1][2] = 0.7*max(Qmf[3]) + 0.3*max(Qmf[2]);

                            # Mix MF and MB
                            Q[2:3] = Qmf[2:3];
                            Q[1] = w[i]*Qmb[1] + (1-w[i])*Qmf[1];
                        }
                    }
                }
            """

        elif model == 'lr_cr_et_w_p':
            self.paramnames = {
                'long' : ['Learning Rate',
                          'Choice Randomness',
                          'Eligibility Trace',
                          'Model-Based Weight',
                          'Perseveration'],
                'code' : ['lr', 'cr', 'et', 'w', 'p']
            }
            self.model = """
                data {
                    int<lower=1> N;
                    int<lower=1> T;
                    int<lower=1,upper=2> S2[T, N];
                    int<lower=1,upper=2> A1[T, N];
                    int<lower=1,upper=2> A2[T, N];
                    real R[T, N];
                }
                transformed data {
                    vector[2] Q_o[3];

                    for (i in 1:3){
                        Q_o[i] = rep_vector(0.0, 2);
                    }

                }
                parameters {
                    # Initialize group-level hyperparameters
                    vector[5] mu_p;
                    vector<lower=0>[5] sigma;

                    # Subject-level parameters
                    vector[N] lr_pr;
                    vector[N] cr_pr;
                    vector[N] et_pr;
                    vector[N] w_pr;
                    vector[N] persev_pr;
                }
                transformed parameters {
                    # Subject-level parameters
                    vector<lower=0, upper=1>[N] lr;
                    vector<lower=0, upper=10>[N] cr;
                    vector<lower=0, upper=1>[N] et;
                    vector<lower=0, upper=1>[N] w;
                    vector[N] persev;

                    for (i in 1:N) {
                        lr[i] = Phi_approx(mu_p[1] + sigma[1]*lr_pr[i]);
                        cr[i] = Phi_approx(mu_p[2] + sigma[2]*cr_pr[i])*10;
                        et[i] = Phi_approx(mu_p[3] + sigma[3]*et_pr[i]);
                        w[i] = Phi_approx(mu_p[4] + sigma[4]*w_pr[i]);
                        persev[i] = mu_p[5] + sigma[5]*persev_pr[i];
                    }
                }
                model {
                    # Hyperparameters
                    mu_p ~ normal(0, 1);
                    sigma ~ cauchy(0, 5);

                    # Individual parameters
                    lr_pr ~ normal(0, 1);
                    cr_pr ~ normal(0, 1);
                    et_pr ~ normal(0, 1);
                    w_pr ~ normal(0, 1);
                    persev_pr ~ normal(0, 1);

                    # Subject and trial loops
                    for (i in 1:N) {
                        vector[2] Q[3];
                        vector[2] Qmf[3];
                        vector[2] Qmb[3];
                        real PE;
                        int a_last;
                        real repr;

                        Q = Q_o;
                        Qmf = Q_o;
                        Qmb = Q_o;

                        a_last = 100;
                        for (t in 1:T) {
                            # Compute perseveration function
                            if (A1[t, i] == a_last)
                              repr = 1;
                            else
                              repr = 0;

                            # First state action
                            A1[t, i] ~ categorical_logit(cr[i]*Q[1] + persev[i]*repr);

                            # Second state action
                            A2[t, i] ~ categorical_logit(cr[i]*Q[S2[t, i]]);

                            # Prediction error
                            PE = R[t, i] - Q[S2[t, i]][A2[t, i]];

                            # Learning (Model Free)
                            Qmf[1][A1[t, i]] = Qmf[1][A1[t, i]] + lr[i]*(Qmf[S2[t, i]][A2[t, i]] - Qmf[1][A1[t, i]]);
                            Qmf[S2[t, i]][A2[t, i]] = Qmf[S2[t, i]][A2[t, i]] + lr[i]*PE;
                            Qmf[1][A1[t, i]] = Qmf[1][A1[t, i]] + lr[i]*PE*et[i];

                            # Learning (Model Based)
                            Qmb[1][1] = 0.7*max(Qmf[2]) + 0.3*max(Qmf[3]);
                            Qmb[1][2] = 0.7*max(Qmf[3]) + 0.3*max(Qmf[2]);

                            # Mix MF and MB
                            Q[2:3] = Qmf[2:3];
                            Q[1] = w[i]*Qmb[1] + (1-w[i])*Qmf[1];
                        }
                    }
                }
            """
# ------------------------------------------------------------------------------
#
#   Bandit
#
# ------------------------------------------------------------------------------

class bandit(GenerativeModel):
    def __init__(self, model='lr_cr'):

        if model == 'lr_cr':
            self.paramnames = {
                'long' : ['Learning Rate',
                          'Choice Randomness'],
                'code' : ['lr', 'cr']
            }
            self.model = """
                data {
                    int<lower=1> N;
                    int<lower=1> T;
                    int<lower=1,upper=2> A[T, N];
                    real R[T, N];
                }
                transformed data {
                    vector[2] Q_o;
                    Q_o = rep_vector(0.0, 2);
                }
                parameters {
                    vector[2] mu_p;
                    vector<lower=0>[2] sigma;

                    vector[N] lr_pr;
                    vector[N] cr_pr;
                }
                transformed parameters {
                    vector<lower=0, upper=1>[N] lr;
                    vector<lower=0, upper=10>[N] cr;

                    for (i in 1:N) {
                        lr[i] = Phi_approx(mu_p[1] + sigma[1]*lr_pr[i]);
                        cr[i] = Phi_approx(mu_p[2] + sigma[2]*cr_pr[i])*10;
                    }
                }
                model {
                    # Hyperparameters
                    mu_p ~ normal(0, 1);
                    sigma ~ cauchy(0, 5);

                    # Individual parameters
                    lr_pr ~ normal(0, 1);
                    cr_pr ~ normal(0, 1);

                    # Subject and trial loops
                    for (i in 1:N) {
                        vector[2] Q;
                        real PE;

                        Q = Q_o;

                        for (t in 1:T) {
                            # Action probability
                            A[t, i] ~ categorical_logit(cr[i]*Q);

                            # Prediction error
                            PE = R[t, i] - Q[A[t, i]];

                            # Learning
                            Q[A[t, i]] = Q[A[t, i]] + lr[i]*PE;
                        }
                    }
                }
            """

        if model == 'lr_cr_rs':
            self.paramnames = {
                'long' : ['Learning Rate',
                          'Choice Randomness',
                          'Reward Sensitivity'],
                'code' : ['lr', 'cr', 'rs']
            }
            self.model = """
                data {
                    int<lower=1> N;
                    int<lower=1> T;
                    int<lower=1,upper=2> A[T, N];
                    real R[T, N];
                }
                transformed data {
                    vector[2] Q_o;
                    Q_o = rep_vector(0.0, 2);
                }
                parameters {
                    vector[3] mu_p;
                    vector<lower=0>[3] sigma;

                    vector[N] lr_pr;
                    vector[N] cr_pr;
                    vector[N] rs_pr;
                }
                transformed parameters {
                    vector<lower=0, upper=1>[N] lr;
                    vector<lower=0, upper=10>[N] cr;
                    vector<lower=0, upper=1>[N] rs;

                    for (i in 1:N) {
                        lr[i] = Phi_approx(mu_p[1] + sigma[1]*lr_pr[i]);
                        cr[i] = Phi_approx(mu_p[2] + sigma[2]*cr_pr[i])*10;
                        rs[i] = Phi_approx(mu_p[1] + sigma[1]*rs_pr[i]);
                    }
                }
                model {
                    # Hyperparameters
                    mu_p ~ normal(0, 1);
                    sigma ~ cauchy(0, 5);

                    # Individual parameters
                    lr_pr ~ normal(0, 1);
                    cr_pr ~ normal(0, 1);
                    rs_pr ~ normal(0, 1);

                    # Subject and trial loops
                    for (i in 1:N) {
                        vector[2] Q;
                        real PE;

                        Q = Q_o;

                        for (t in 1:T) {
                            # Action probability
                            A[t, i] ~ categorical_logit(cr[i]*Q);

                            # Prediction error
                            PE = rs[i]*R[t, i] - Q[A[t, i]];

                            # Learning
                            Q[A[t, i]] = Q[A[t, i]] + lr[i]*PE;
                        }
                    }
                }
            """
