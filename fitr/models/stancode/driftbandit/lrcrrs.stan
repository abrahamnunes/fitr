data {
    int<lower=2> K;
    int<lower=1> N;
    int<lower=1> T;
    int<lower=1,upper=2> A[T, N];
    real R[T, N];
}
transformed data {
    vector[K] Q_o;
    Q_o = rep_vector(0.0, K);
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
        vector[K] Q;
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
