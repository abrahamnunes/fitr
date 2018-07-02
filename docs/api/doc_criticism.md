# `fitr.criticism`

Methods for criticism of model fits.



## actual_estimate

```python
fitr.criticism.plotting.actual_estimate(y_true, y_pred, xlabel='Actual', ylabel='Estimate', corr=True, figsize=None)
```

Plots parameter estimates against the ground truth values.

Arguments:

- **y_true**: `ndarray(nsamples)`. Vector of ground truth parameters
- **y_pred**: `ndarray(nsamples)`. Vector of parameter estimates
- **xlabel**: `str`. Label for x-axis
- **ylabel**: `str`. Label for y-axis
- **corr**: `bool`. Whether to plot correlation coefficient.
- **figsize**: `tuple`. Figure size (inches).

Returns:

`matplotlib.pyplot.Figure`

---


