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
    vector[2] p_vec_o;

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
    vector[N] persev_pr;
    vector[N] w_pr;
}
transformed parameters {
    # Subject-level parameters
    vector<lower=0, upper=1>[N] lr;
    vector<lower=0, upper=10>[N] cr;
    vector<lower=-1, upper=1>[N] persev;
    vector<lower=0, upper=1>[N] w;

    for (i in 1:N) {
        lr[i] = Phi_approx(mu_p[1] + sigma[1]*lr_pr[i]);
        cr[i] = Phi_approx(mu_p[2] + sigma[2]*cr_pr[i])*10;
        persev[i] = mu_p[3] + sigma[3]*persev_pr[i];
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
    w_pr ~ normal(0, 1);
    persev_pr ~ normal(0, 1);

    # Subject and trial loops
    for (i in 1:N) {
        vector[2] Q[3];
        vector[2] Qmf[3];
        vector[2] Qmb[3];
        vector[2] p_vec;
        real PE;

        Q = Q_o;
        Qmf = Q_o;
        Qmb = Q_o;

        p_vec = p_vec_o;

        for (t in 1:T) {
            # First state action
            A1[t, i] ~ categorical_logit(cr[i]*(Q[1] + persev[i]*p_vec));

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

            for (j in 1:2) {
              if (A1[t, i] == j) {
                p_vec[j] = 1;
              } else {
                p_vec[j] = 0;
              }
            }
        }
    }
}
