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
import fitr.utils as fu

def logsumexp(x):
    """ Gradient for the logsumexp function taken at point $x$:

    $$
    \\frac{\\partial}{\\partial x} \\log \\sum_i e^{x_i} = \Big( \\frac{e^{x_0}}{\\sum_i e^{x_i}}, \ldots, \\frac{e^{x_j}}{\\sum_i e^{x_i}}, \ldots, \\frac{e^{x_n}}{\\sum_i e^{x_i}} \Big)^\\top
    $$

    Arguments:

        x: `ndarray((n,))`

    Returns:

        `ndarray((n,))`

    """
    return fu.softmax(x)

def max(x):
    """ Gradient of a max reduction over a vector, elementwise

    Arguments:

        x: `ndarray((n,))`

    Returns:

        `ndarray((n,))`

    """
    xmax = np.max(x)
    out = np.zeros(x.size)
    wheremax = np.equal(x, xmax)
    nmax = wheremax.astype(np.int).sum()
    out[wheremax] = 1/nmax
    return out

def matrix_max(X, axis=0):
    """ Gradient of a max reduction over a matrix's rows or columns, elementwise

    Arguments:

        X: `ndarray((n,m))`. Matrix to which the `max` operation is applied
        axis: `int`. Over which axis the `max` operation is applied

    Returns:

        `ndarray((n,m))`

    """
    if axis == 0:
        G = np.hstack(max(X[:,j]).reshape(-1, 1) for j in range(X.shape[1]))
    elif axis == 1:
        G = np.vstack(max(X[i,:]).reshape(1, -1) for i in range(X.shape[0]))
    return G

def softmax(x):
    """ Jacobian of the softmax function.

    Let

        - $x = (x_0, x_1, \ldots, x_n)^\\top
        - $v = (e^{x_0}, e^{x_1}, \ldots, e^{x_n})^\\top$
        - $z = \sum_{i=0}^n v_i
        - $\\mathbf{I}$ be the identity matrix of size $n \\times n$.

    Then the Jacobian of the softmax function $\\varsigma(x)$ is

    $$
    \\partial_x \\varsigma(x) = \\frac{zv \\mathbf I - vv^\\top}{z^2}
    $$

    Arguments:

        x: `ndarray((n,))`

    Returns:

        `ndarray((n,n))`

    """
    x = x - np.max(x)
    v = np.exp(x)
    z = np.sum(v)
    return (np.diag(z*v) - np.outer(v, v))/(z**2)

def exp(x):
    """ Derivative of exponential function

    Trivial, but here for consistency.

    Arguments:

        x: `ndarray`. Vector of inputs

    Returns:

        Returns `x` unchanged
    """
    return x

def sigmoid(x, a_min=-10, a_max=10):
    """ Derivative of sigmoid functionself.

    The sigmoid function is

    $$
    \\sigma(x) = \\frac{1}{1+e^{-x}}
    $$

    and its derivative is

    $$
    \\partial_x \\sigma(x) = \\frac{e^{-x}}{(1+e^{-x})^2}.
    $$

    Arguments:

        x: `float` or `ndarray`. Inputs to the sigmoid
        a_min: Lower bound at which to clip values of `x`
        a_max: Upper bound at which to clip values of `x`

    Returns:

        `float` or `ndarray(shape=x.size)`

    """
    v = np.exp(-x)
    return v/((1 + v)**2)
