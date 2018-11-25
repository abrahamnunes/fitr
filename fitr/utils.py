# -*- coding: utf-8 -*-
import scipy.io as spio
import autograd.numpy as np

def batch_softmax(X, axis=1):
    """ Computes the softmax function for a batch of samples

    $$
    p(\mathbf{x}) = \\frac{e^{\mathbf{x} - \max_i x_i}}{\mathbf{1}^\\top e^{\mathbf{x} - \max_i x_i}}
    $$

    Arguments:

        x: Softmax logits (`ndarray((nsamples,nfeatures))`)

    Returns:

        Matrix of probabilities of size `ndarray((nsamples,nfeatures))` such that sum over `nfeatures` is 1.
    """
    xmax = reduce_then_tile(X, np.max, axis=axis)
    expx = np.exp(X - xmax)
    y = expx/reduce_then_tile(expx, np.sum, axis=axis)
    return y

def batch_transform(X, f_list):
    """ Applies the `fitr.utils.transform` function over a batch of parameters

    Arguments:

        X: `ndarray((nsamples, nparams))`. Raw parameters
        f_list: `list` where `len(list) == nparams`. Functions defining coordinate transformations on each element of `x`.

    Returns:

        `ndarray((nsamples, nparams))`. Transformed parameters
    """
    return np.stack(np.ravel(transform(X[i], f_list)) for i in range(X.shape[0]))


def bitflip(x):
    """ Flips the bits of a binary vector
    
    
    Arguments: 

        x: `ndarray((n, m))`. Only meaningful if binary. 

    Returns: 

        `ndarray(x.shape)`. Zeros become 1 and ones become 0.

    """
    return signinv(-sign(x))


def I(x):
    """ Identity transformation.

    Mainly for convenience when using `fitr.utils.transform` with some vector element that should not be transformed, despite changing the coordinates of other variables.

    Arguments:

        x: `ndarray`

    Returns:

        `ndarray(shape=x.shape)`

    """
    return x


def log_loss(p, q):
    """ Computes log loss.

    $$
    \mathcal L = - \\frac{1}{n_s} \\big( \mathbf p^\\top \log \mathbf q + (1-\mathbf p)^\\top \log (1 - \mathbf q) \\big)
    $$

    Arguments:

        p: Binary vector of true labels `ndarray((nsamples,))`
        q: Vector of estimates (between 0 and 1) of type `ndarray((nsamples,))`

    Returns:

        Scalar log loss
    """
    return -np.mean(p*np.log(q) + (1-p)*np.log(1-q))

def logsumexp(x):
    """ Numerically stable logsumexp.

    Computed as follows:

    $$
    \max x + \log \sum_x e^{x - \max x}
    $$

    Arguments:

        x: `ndarray(shape=(nactions,))``

    Returns:

        `float`
    """
    xmax = np.max(x)
    y = xmax + np.log(np.sum(np.exp(x-xmax)))
    return y 

def make_onehot(x):
    """ Turns a vector of labels into a one-hot array

    Arguments:

        x: `ndarray(nsamples)`. Group labels

    Returns:

        xout: `ndarray((nsamples, ngroups))`. Data as onehot vectors
        labels: `ndarray(ngroups)`. Group labels
    """
    nsamples = x.size
    labels = np.unique(x)
    ngroups = labels.size
    xout = np.zeros((nsamples, ngroups))
    for i, l in enumerate(labels):
        xout[x == l, i] = 1

    return xout, labels

def getquantile(x, lower=0.025, upper=0.975, return_indices=False):
    """ Indicates which elements of `x` fall into a quantile range
    
    Arguments: 

        x: `ndarray(nsamples)`
        lower: `0<=float<max(upper,1)`. Lower quantile
        upper: `min(0, lower)<float<=1`. Upper quantile
        return_indices: `bool`. If `False`, returns boolean array. If `True` returns indices for entries of `x` falling between `lower` and `upper`.
    
    Returns: 

        `ndarray`. Dimensionality will depend on `return_indices`

    """
    lb, ub = np.percentile(x, [lower*100, upper*100])
    y = np.logical_and(np.greater_equal(x, lb), np.less(x, ub))

    if return_indices: 
        y = np.arange(x.size)[y]

    return y

def rank_data(x):
    """ Ranks a set of observations, assigning the average of ranks to ties. 


    Arguments:

        x: `ndarray(nsamples)`. Vector of data to be compared

    Returns:

        ranks: `ndarray(nsamples)`. Ranks for each observation
    
    """
    x = x.flatten()
    nsamples = x.size

    # Sort in ascenting order
    idx = np.argsort(x)
    ranks = np.empty(idx.size)    
    ranks[idx] = np.arange(idx.size) + 1

    # Now average the ranks for ties
    unique_x = np.unique(x)
    if unique_x.size < nsamples:
        for i, xi in enumerate(unique_x):
            if x[x == xi].size > 1:
                ranks[x==xi] = np.mean(ranks[x == xi])

    return ranks


def rank_grouped_data(x, g):
    """ Ranks observations taken across several groups

    Arguments:

        x: `ndarray(nsamples)`. Vector of data to be compared
        g: `ndarray(nsamples)`. Group ID's

    Returns:

        ranks: `ndarray(nsamples)`. Ranks for each observation
        G: `ndarray(nsamples, ngroups)`.  Matrix indicating whether sample i is in group j
        R: `ndarray((nsamples, ngroups))`. Matrix indicating the rank for sample i in group j
        lab: `ndarray(ngroups)`. Group labels
    """
    nsamples = x.size
    ngroups = np.unique(g).size

    # Sort in ascending order
    idx   = np.argsort(x)
    G,lab = make_onehot(g[idx])

    ranks = rank_data()    

    R = np.tile(ranks.reshape(-1, 1), [1, ngroups]) * G
    return ranks, G, R, lab


def reduce_then_tile(X, f, axis=1):
    """ Computes some reduction function over an axis, then tiles that vector to create matrix of original size

    Arguments:

        X: `ndarray((n, m))`. Matrix.
        f: `function` that reduces data across some axis (e.g. `np.sum()`, `np.max()`)
        axis: `int` which axis the data should be reduced over (only goes over 2 axes for now)

    Returns:res

        `ndarray((n, m))`

    Examples:

    Here is one way to compute a softmax function over the columns of `X`, for each row.

    ```
    import numpy as np
    X = np.random.normal(0, 1, size=(10, 3))**2
    max_x = reduce_then_tile(X, np.max, axis=1)
    exp_x = np.exp(X - max_x)
    sum_exp_x = reduce_then_tile(exp_x, np.sum, axis=1)
    y = exp_x/sum_exp_x
    ```

    """
    y = f(X, axis=axis)
    if axis==1:
        y = np.tile(y.reshape(-1, 1), [1, X.shape[1]])
    elif axis==0:
        y = np.tile(y.reshape(1, -1), [X.shape[0], 1])
    return y

def relu(x, a_max=None):
    """ Rectified linearity

    $$
    \mathbf x' = \max (x_i, 0)_{i=1}^{|\mathbf x|}
    $$

    Arguments:

        x: Vector of inputs
        a_max: Upper bound at which to clip values of `x`

    Returns:

        Exponentiated values of `x`.

    """
    if a_max is None: a_max=np.inf
    x = x.clip(max=a_max)
    return np.greater(x, 0)*x

def scale_data(X, axis=0, with_mean=True, with_var=True, copy=True):
    """ Rescales data by subtracting mean and dividing by standard deviation. 

    $$
    \mathbf x' = \\frac{\mathbf x - \\frac{1}{n} \mathbf 1^\\top \mathbf x}{SD(\mathbf x)}
    $$

    Arguments:

        X: `ndarray((nsamples, [nfeatures]))`. Data. May be 1D or 2D.
        axis: `int`. Over which axis to scale
        with_mean: `bool`. Whether to subtract the mean
        with_var: `bool`. Whether to normalize for variance
        copy: `bool`. Copies array so values are not normalized in place 

    Returns:

        `ndarray(X.shape)`. Rescaled data.
    """
    if copy: 
        X = X.copy()

    if X.ndim == 1: 
        X = X.reshape(-1, 1)
    
    if with_mean: 
        X -= reduce_then_tile(X, np.mean, axis)
    if with_var: 
        xstd = reduce_then_tile(X, np.std, axis)
        xstd[xstd == 0] = 1.
        X /= xstd
    return X

def sigmoid(x, a_min=-10, a_max=10):
    """ Sigmoid function

    $$
    \sigma(x) = \\frac{1}{1 + e^{-x}}
    $$

    Arguments:

        x: Vector
        a_min: Lower bound at which to clip values of `x`
        a_max: Upper bound at which to clip values of `x`

    Returns:

        Vector between 0 and 1 of size `x.shape`
    """
    expnx = np.exp(-np.clip(x, a_min=a_min, a_max=a_max))
    return 1/(1+expnx)

def sign(x):
    """ Sign function (converts 0, 1 to -1, 1)

    Arguments: 

        x: `ndarray((n, m))`. Only meaningful if binary. 

    Returns: 

        `ndarray(x.shape)`. Zeros become -1 and ones become 1.

    """
    return 2*x -1

def signinv(x):
    """ Inverse of sign function (converts -1, 1 to 0, 1)

    Arguments: 

        x: `ndarray((n, m))`. Only meaningful if binary. 

    Returns: 

        `ndarray(x.shape)`. Zeros become -1 and ones become 1.

    """
    return (x + 1)/2

def softmax(x):
    """ Computes the softmax function

    $$
    p(\mathbf{x}) = \\frac{e^{\mathbf{x} - \max_i x_i}}{\mathbf{1}^\\top e^{\mathbf{x} - \max_i x_i}}
    $$

    Arguments:

        x: Softmax logits (`ndarray((N,))`)

    Returns:

        Vector of probabilities of size `ndarray((N,))`
    """
    xmax = np.max(x)
    expx = np.exp(x-xmax)
    return expx/np.sum(expx)

def softmax_components(x):
    """ Returns the potential (numerator) and partition function (denominator) of softmax

    Given a vector $\mathbf x = (x_1, x_2, \ldots, x_n)$ he potential of the softmax function is

    $$
    \psi(\mathbf x) = e^{\mathbf x}
    $$

    The partition function for the softmax is

    $$
    \eta(\mathbf x) = \sum_{i} \psi(x_i)
    $$

    Arguments:

        x: Softmax logits (`ndarray((n,))`)

    Returns:

        potential: `ndarray(x.size)`
        partition: `float`
    """
    xmax = np.max(x)
    potential = np.exp(x-xmax)
    partition = np.sum(potential)
    return potential, partition

def stable_exp(x, a_min=-10, a_max=10):
    """ Clipped exponential function

    Avoids overflow by clipping input values.

    Arguments:

        x: Vector of inputs
        a_min: Lower bound at which to clip values of `x`
        a_max: Upper bound at which to clip values of `x`

    Returns:

        Exponentiated values of `x`.
    """
    return np.exp(np.clip(x, a_min=a_min, a_max=a_max))

def tanh(x, a_min=-10, a_max=10):
    """ Hyperbolic tangent function

    $$
    \tanh(x) = \\frac{e^x-e^{-x}}{e^x + e^{-x}}
    $$

    Arguments:

        x: `ndarray((n, m))`. Values to transform
        a_min: `int`. Lower bound at which to clip values of `x`
        a_max: `int`. Upper bound at which to clip values of `x`

    Returns:

        `ndarray(x.shape)`. Transformed values between -1 and 1 of size `x.shape`
    """
    tanhvals = np.clip(x, a_min=a_min, a_max=a_max)
    return np.tanh(tanhvals)

def transform(x, f_list):
    """ Transforms parameters from domain in `x` into some new domain defined by `f_list`

    Arguments:

        x: `ndarray((nparams,))`. Parameter vector in some domain.
        f_list: `list` where `len(list) == nparams`. Functions defining coordinate transformations on each element of `x`.

    Returns:

        x_: `ndarray((nparams,))`. Parameter vector in new coordinates.

    Examples:

    Applying `fitr` transforms can be done as follows.

    ``` python
    import numpy as np
    from fitr.utils import transform, sigmoid, relu

    x = np.random.normal(0, 5, size=3)
    x_= transform(x, [sigmoid, relu, relu])
    ```

    You can also apply other functions, so long as dimensions are equal for input and output.

    ``` python
    import numpy as np
    from fitr.utils import transform

    x  = np.random.normal(0, 10, size=3)
    x_ = transform(x, [np.square, np.sqrt, np.exp])
    ```
    """
    if x.ndim == 1:
        x = np.stack(np.array([x_i]) for i, x_i in enumerate(x))
    x_ = np.stack(f_list[i](x_i) for i, x_i in enumerate(x))
    return x_

# ============================================================================
#   LOADMAT FUNCTION AND RELATED METHODS 
# ============================================================================

def _todict(data):
    """ A helper function for the `loadmat` function """
    dout = {}
    for s in data._fieldnames:
        element = data.__dict__[s]
        if isinstance(element, spio.matlab.mio5_params.mat_struct):
            dout[s] = _todict(element)
        else: 
            dout[s] = element
    return dout

def _check_keys(data):
    """ A helper function for the `loadmat` function """
    for key in data:
        if isinstance(data[key], spio.matlab.mio5_params.mat_struct):
            data[key] = _todict(data[key])
    return data

def loadmat(fname):
    """ Loads a `.mat` file and parses it to make it easy to work win in python. 

    This code was taken largely from the stackoverflow post at \href{https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries}{}https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries}.
    
    Arguments: 

        fname: `str`. File name.

    """ 
    data = spio.loadmat(fname, 
                        struct_as_record=False,
                        squeeze_me=True) 
    return _check_keys(data)
