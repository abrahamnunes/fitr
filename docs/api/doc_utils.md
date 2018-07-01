# `fitr.utils`

Functions used across `fitr`.



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



## log_loss

```python
fitr.utils.log_loss(p, q)
```

Log-loss function.

$$
\mathcal L = \mathbf p^\top \log \mathbf q + (1-\mathbf p)^\top \log (1 - \mathbf q)
$$

Arguments:

- **p**: Binary vector of true labels `ndarray((nsamples,))`
- **q**: Vector of estimates (between 0 and 1) of type `ndarray((nsamples,))`

Returns:

Scalar log loss

---


