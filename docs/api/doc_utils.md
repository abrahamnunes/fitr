# `fitr.utils`

Functions used across `fitr`.



## batch_softmax

```python
fitr.utils.batch_softmax(X, axis=1)
```

Computes the softmax function for a batch of samples

$$
p(\mathbf{x}) = \frac{e^{\mathbf{x} - \max_i x_i}}{\mathbf{1}^\top e^{\mathbf{x} - \max_i x_i}}
$$

Arguments:

- **x**: Softmax logits (`ndarray((nsamples,nfeatures))`)

Returns:

Matrix of probabilities of size `ndarray((nsamples,nfeatures))` such that sum over `nfeatures` is 1.

---



## batch_transform

```python
fitr.utils.batch_transform(X, f_list)
```

Applies the `fitr.utils.transform` function over a batch of parameters

Arguments:

- **X**: `ndarray((nsamples, nparams))`. Raw parameters
- **f_list**: `list` where `len(list) == nparams`. Functions defining coordinate transformations on each element of `x`.

Returns:

`ndarray((nsamples, nparams))`. Transformed parameters

---



## I

```python
fitr.utils.I(x)
```

Identity transformation.

Mainly for convenience when using `fitr.utils.transform` with some vector element that should not be transformed, despite changing the coordinates of other variables.

Arguments:

- **x**: `ndarray`

Returns:

`ndarray(shape=x.shape)`

---



## log_loss

```python
fitr.utils.log_loss(p, q)
```

Computes log loss.

$$
\mathcal L = - \frac{1}{n_s} \big( \mathbf p^\top \log \mathbf q + (1-\mathbf p)^\top \log (1 - \mathbf q) \big)
$$

Arguments:

- **p**: Binary vector of true labels `ndarray((nsamples,))`
- **q**: Vector of estimates (between 0 and 1) of type `ndarray((nsamples,))`

Returns:

Scalar log loss

---



## logsumexp

```python
fitr.utils.logsumexp(x)
```

Numerically stable logsumexp.

Computed as follows:

$$
\max x + \log \sum_x e^{x - \max x}
$$

Arguments:

- **x**: `ndarray(shape=(nactions,))``

Returns:

`float`

---



## rank_data

```python
fitr.utils.rank_data(x)
```

Ranks a set of observations, assigning the average of ranks to ties. 


Arguments:

- **x**: `ndarray(nsamples)`. Vector of data to be compared

Returns:

- **ranks**: `ndarray(nsamples)`. Ranks for each observation

---



## rank_grouped_data

```python
fitr.utils.rank_grouped_data(x, g)
```

Ranks observations taken across several groups

Arguments:

- **x**: `ndarray(nsamples)`. Vector of data to be compared
- **g**: `ndarray(nsamples)`. Group ID's

Returns:

- **ranks**: `ndarray(nsamples)`. Ranks for each observation
- **G**: `ndarray(nsamples, ngroups)`.  Matrix indicating whether sample i is in group j
- **R**: `ndarray((nsamples, ngroups))`. Matrix indicating the rank for sample i in group j
- **lab**: `ndarray(ngroups)`. Group labels

---



## reduce_then_tile

```python
fitr.utils.reduce_then_tile(X, f, axis=1)
```

Computes some reduction function over an axis, then tiles that vector to create matrix of original size

Arguments:

- **X**: `ndarray((n, m))`. Matrix.
- **f**: `function` that reduces data across some axis (e.g. `np.sum()`, `np.max()`)
- **axis**: `int` which axis the data should be reduced over (only goes over 2 axes for now)

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

---



## relu

```python
fitr.utils.relu(x, a_max=None)
```

Rectified linearity

$$
\mathbf x' = \max (x_i, 0)_{i=1}^{|\mathbf x|}
$$

Arguments:

- **x**: Vector of inputs
- **a_max**: Upper bound at which to clip values of `x`

Returns:

Exponentiated values of `x`.

---



## scale_data

```python
fitr.utils.scale_data(X, axis=0, with_mean=True, with_var=True)
```

Rescales data by subtracting mean and dividing by standard deviation. 

$$
\mathbf x' = \frac{\mathbf x - \frac{1}{n} \mathbf 1^\top \mathbf x}{SD(\mathbf x)}
$$

Arguments:

- **X**: `ndarray((nsamples, [nfeatures]))`. Data. May be 1D or 2D.
- **with_mean**: `bool`. Whether to subtract the mean
- **with_var**: `bool`. Whether to normalize for variance

Returns:

`ndarray(X.shape)`. Rescaled data.

---



## sigmoid

```python
fitr.utils.sigmoid(x, a_min=-10, a_max=10)
```

Sigmoid function

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

Arguments:

- **x**: Vector
- **a_min**: Lower bound at which to clip values of `x`
- **a_max**: Upper bound at which to clip values of `x`

Returns:

Vector between 0 and 1 of size `x.shape`

---



## softmax

```python
fitr.utils.softmax(x)
```

Computes the softmax function

$$
p(\mathbf{x}) = \frac{e^{\mathbf{x} - \max_i x_i}}{\mathbf{1}^\top e^{\mathbf{x} - \max_i x_i}}
$$

Arguments:

- **x**: Softmax logits (`ndarray((N,))`)

Returns:

Vector of probabilities of size `ndarray((N,))`

---



## stable_exp

```python
fitr.utils.stable_exp(x, a_min=-10, a_max=10)
```

Clipped exponential function

Avoids overflow by clipping input values.

Arguments:

- **x**: Vector of inputs
- **a_min**: Lower bound at which to clip values of `x`
- **a_max**: Upper bound at which to clip values of `x`

Returns:

Exponentiated values of `x`.

---



## transform

```python
fitr.utils.transform(x, f_list)
```

Transforms parameters from domain in `x` into some new domain defined by `f_list`

Arguments:

- **x**: `ndarray((nparams,))`. Parameter vector in some domain.
- **f_list**: `list` where `len(list) == nparams`. Functions defining coordinate transformations on each element of `x`.

Returns:

- **x_**: `ndarray((nparams,))`. Parameter vector in new coordinates.

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

---


