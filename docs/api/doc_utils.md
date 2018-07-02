# `fitr.utils`

Functions used across `fitr`.



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

Rescales data by subtracting mean and dividing by variance

$$
\mathbf x' = \frac{\mathbf x - \frac{1}{n} \mathbf 1^\top \mathbf x}{Var(\mathbf x)}
$$

Arguments:

- **X**: `ndarray((nsamples, [nfeatures]))`. Data. May be 1D or 2D.
- **with_mean**: `bool`. Whether to subtract the mean
- **with_var**: `bool`. Whether to divide by variance

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


