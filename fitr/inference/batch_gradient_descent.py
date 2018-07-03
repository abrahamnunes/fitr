# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

'''
def loglik(W, D):
    """
    w = Tensor((batch_size, n_params)) # [alpha, beta, omega]
    D = Tensor((batch_size, time_steps, timestep_data_size))
    """
    batch_size, ntrials, _ = D.shape.as_list()
    nstates=4; nactions=2

    alpha = tf.nn.sigmoid(tf.squeeze(tf.slice(w, [0, 2], [batch_size, 1])))
    omega = tf.nn.sigmoid(tf.squeeze(tf.slice(w, [0, 2], [batch_size, 1])))

    # Create log-likelihood running value
    L = tf.zeros([batch_size])

    # Create value function tensor
    Q  = tf.zeros(shape=[batch_size, nactions, nstates])
    u_ = tf.ones([batch_size, nactions])*(1/nactions)

    def cond(t, D, W, Q, u_, L):
        return t < ntrials

    def body(t, D, W, Q, u_, L):
        x  = tf.squeeze(tf.slice(D, [0, t, 0], [batch_size, 1, nstates]))
        u  = tf.squeeze(tf.slice(D, [0, t, nstates], [batch_size, 1, nactions]))
        r  = tf.squeeze(tf.slice(D, [0, t, nstates+nactions], [batch_size, 1, 1]))
        x_ = tf.squeeze(tf.slice(D, [0, t, nstates+nactions+1], [batch_size, 1, nstates]))
        Qx    = tf.einsum('iux,ix->iu', Q,x)
        Qx_   = tf.einsum('iux,ix->iu', Q,x_)
        uQx   = tf.einsum('iu,iu->i', u, Qx)
        u_Qx_ = 0.5*tf.einsum('iu,iu->i', u_, Qx_)
        z     = tf.einsum('iu,ix->iux', u, x) + tf.einsum('iu,ix->iux', u_, x_)
        u_logits = tf.tile(tf.reshape(tf.nn.softplus(tf.squeeze(W[1])), [batch_size, 1]), [1, 2])*Qx
        L  = L - Multinomial(total_count=1., logits=u_logits).log_prob(u)
        dQ = tf.tile(tf.reshape(tf.nn.sigmoid(tf.squeeze(W[0]))*(r-u_Qx_), [batch_size,1,1]), [1, nactions, nstates])
        Q  = Q + dQ*z
        return t+1, D, W, Q, u_, L

    t = tf.constant(0)
    loop_vars = [t, D, W, Q, u_, L]
    t, D, W, Q, u_, L = tf.while_loop(cond, body, loop_vars)

    return L

alpha = tf.Variable(tf.random_normal(mean=0., stddev=2., shape=[N, 1]))
beta = tf.Variable(tf.random_normal(mean=0., stddev=2., shape=[N, 1]))
'''

def batch_gradient_descent(f, X, param_list):
    """
    Learns parameters of reinforcement learning model `f` simultaneously over all subjects represented in behavioural data tensor

    Parameters
    ----------
    f
        Log-likelihood function. See page on ``How to build a log-likelihood function'' for details and requirements.
    X : ndarray((nsubjects, ntrials, nfeatures))
        Behavioural data tensor. `nfeatures` is the length of the flattened $x,u,r,x',u'$ vector (or equivalent)
    param_list : list
        List of tensorflow variables defining the initial conditions for variables

    Returns
    -------
    OptimizationResult
    """
    N, T, M = X.shape
    X_ = tf.constant(X.astype(np.float32), dtype=tf.float32)


    L = loglik(param_list, X)
    J = tf.reduce_sum(L)
    train = tf.train.AdamOptimizer(1e-1).minimize(J)

    LOSS = []
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    converged = False; t = 0;
    lr0 = sess.run(alpha)
    cr0 = sess.run(beta)
    while not converged:
        sess.run(train)
        loss = sess.run(J)
        LOSS.append(loss)
        lr = sess.run(alpha)
        cr = sess.run(beta)
        lrfro = np.linalg.norm(lr-lr0, ord='fro')
        crfro = np.linalg.norm(cr-cr0, ord='fro')
        lr0 = lr
        cr0 = cr
        if t % 10 == 0:
            print('Iter %s Loss %s LRfro %s CRfro %s' %(t, loss, lrfro, crfro))
        t += 1
        if t > 2 and np.abs(LOSS[-2]-LOSS[-1]) < 1e-2:
            converged = True
    lr = (sess.run(alpha)).flatten()
    cr = (sess.run(beta)).flatten()
