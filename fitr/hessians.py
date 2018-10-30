# -*- coding: utf-8 -*-
# Fitr. A package for fitting reinforcement learning models to behavioural data
# Copyright (C) 2018 Abraham Nunes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# CONTACT INFO:
#   Abraham Nunes
#    Email: nunes@dal.ca
#
# ============================================================================
import numpy as np
import fitr.gradients as grad
import fitr.utils as fu

def log_softmax(B, q):
    """ Hessian for the log-probability of the softmax policy, $\\log p(u|q, \\beta)$ with inverse softmax temperature $\\beta$.

    Let

        - $n_u \\in \\mathbb N_+$ be the dimensionality of the action space
        - $\\beta \\in \\mathbb R_+$ be the inverse softmax temperature
        - $q = (q_1, q_2, \ldots, q_{n_u})^\\top$ be action values
        - $v = (e^{\\beta q_1}, e^{\\beta q_2}, \ldots, e^{\\beta q_{n_u}})^\\top$ be the exponentiated scaled action value
        - $z = \sum_{i=1}^{n_u} v_i$ be the normalization constant

    Then the Hessian with respect to the inverse softmax parameter

    $$
    \\partial^2_\\beta \\log p(u|q, \\beta) = \\Bigg(\\frac{(q_i q_i v^i v^i}{z^2} - \\frac{q_i q_i v^i}{z} \\Bigg)_{k=1}^{n_u}
    $$

    where Einstein summation is operating, and the Hessian with respect to the action values is

    $$
    \\partial^2_q \\log p(u|q, \\beta) = \\frac{\\beta^2}{z^2} v^i v_j - \\frac{\\beta^2}{z} \\mathrm{diag}(v).
    $$

    Arguments:

        B: `float`. Inverse softmax temperature
        q: `ndarray((nactions,))`. Action values

    Returns:

        HB: `ndarray((nactions,))`. Second order partial derivatives of log-softmax probability with respect to inverse temperature
        Hq: `ndarray((nactions, nactions, nactions))`. Second order partial derivatives of log-softmax probability with respect to action values

    """
    q = q - np.max(q)
    v = np.exp(B*q)
    z = np.sum(v)

    # Hessian with respect to the inverse softmax
    HB = np.ones(q.size)*((np.dot(q, v)**2)/(z**2) - np.dot((q**2), v)/z)

    # Hessian with respect to the action values
    Hq = ((B**2)/z)*(np.outer(v, v)/z - np.diag(v))
    Hq = np.tile(np.expand_dims(Hq, 0), [v.size, 1, 1])

    return HB, Hq

def log_stickysoftmax(B, p, q, u):
    """ Hessian for the log-probability of the sticky softmax policy, $\\log p(u'|q, \\beta, \\rho, u)$.

    Let

        - $u' = (u'_0, u'_1, \ldots, u'_{n_u})^\\top$ be the (one-hot) action at the current step
        - $u = (u_0, u_1, \ldots, u_{n_u})^\\top$ be the (one-hot) action at the previous step
        - $n_u \\in \\mathbb N_+$ be the dimensionality of the action space
        - $\\beta \\in \\mathbb R_+$ be the inverse softmax temperature
        - $\\rho \\in \\mathbb R$ be the perseveration parameter
        - $q = (q_1, q_2, \ldots, q_{n_u})^\\top$ be action values
        - $v = (e^{\\beta q_1}, e^{\\beta q_2}, \ldots, e^{\\beta q_{n_u}})^\\top$ be the exponentiated scaled action value
        - $z = \sum_{i=1}^{n_u} v_i$ be the (scalar) normalization constant

    Then the Hessian with respect to the inverse softmax parameter is

    $$
    \\partial^2_\\beta \\log p(u'|q, \\beta, \\rho, u) = \\Bigg(\\frac{(q_i q_i v^i v^i}{z^2} - \\frac{q_i q_i v^i}{z} \\Bigg)_{k=1}^{n_u}
    $$

    where Einstein summation is operating (only the 2 over the $z$ is not an index), and with respect to the perseveration parameter

    $$
    \\partial^2_\\rho \\log p(u'|q, \\beta, \\rho, u) = \\Bigg(\\frac{(u_i u_i v^i v^i}{z^2} - \\frac{u_i u_i v^i}{z} \\Bigg)_{k=1}^{n_u}
    $$

    the Hessian with respect to the action values is

    $$
    \\partial^2_q \\log p(u'|q, \\beta, \\rho, u) = \\frac{\\beta^2}{z^2} v^i v_j - \\frac{\\beta^2}{z} \\mathrm{diag}(v).
    $$

    Arguments:

        B: `float`. Inverse softmax temperature
        p: `float`. Perseveration parameter
        q: `ndarray((nactions,))`. Action values
        u: `ndarray((nactions,))`. One-hot vector representing the action taken at the prior trial

    Returns:

        HB: `ndarray((nactions,))`. Second order partial derivatives of log-(sticky)softmax probability with respect to inverse temperature
        Hp: `ndarray((nactions,))`. Second order partial derivatives of log-(sticky)softmax probability with respect to perseveration parameter
        HBp: `ndarray((nactions,))`. Second order partial derivative of log-(sticky)softmax probability with respect to inverse temperature and perseveration parameter
        Hq: `ndarray((nactions, nactions, nactions))`. Second order partial derivatives of log-(sticky)softmax probability with respect to action values
        Hu: `ndarray((nactions, nactions, nactions))`. Second order partial derivatives of log-(sticky)softmax probability with respect to previous action

    """
    q = q - np.max(q)
    u = u - np.max(u)
    v = np.exp(B*q + p*u)
    z = np.sum(v)

    # Second-order partial derivative with respect to the inverse softmax
    HB = (np.dot(q, v)**2)/(z**2) - np.dot((q**2), v)/z
    HB = np.ones(q.size)*HB

    # Second-order partial derivative with respect to the perseveration parameter
    Hp = (np.dot(u, v)**2)/(z**2) - np.dot((u**2), v)/z
    Hp = np.ones(u.size)*Hp

    # Second-order partial derivative with respect to the inverse softmax and perseveration parameters
    HBp = -u@grad.softmax(B*q + p*u)@q
    HBp = np.ones(u.size)*HBp

    # Second-order partial derivative with respect to the action values
    Vouter = np.outer(v, v)
    Vdiag = np.diag(v)
    Hq = ((B**2)/z)*(Vouter/z - Vdiag)
    Hq = np.tile(np.expand_dims(Hq, 0), [v.size, 1, 1])

    # Second-order partial derivative with respect to the last action
    Hu = ((p**2)/z)*(Vouter/z - Vdiag)
    Hu = np.tile(np.expand_dims(Hu, 0), [v.size, 1, 1])

    return HB, Hp, HBp, Hq, Hu
