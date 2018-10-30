# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


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
