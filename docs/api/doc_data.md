# `fitr.data`

A module containing a generic class for behavioural data.



## BehaviouralData

```python
fitr.data.BehaviouralData()
```

A flexible and generic object to store and process behavioural data across tasks

Arguments:

- **ngroups**: Integer number of groups represented in the dataset. Only > 1 if data are merged
- **nsubjects**: Integer number of subjects in dataset
- **ntrials**: Integer number of trials done by each subject
- **dict**: Dictionary storage indexed by subject.
- **params**: `ndarray((nsubjects, nparams + 1))` parameters for each (simulated) subject
- **meta**: Array of covariates of type `ndarray((nsubjects, nmetadata_features+1))`
- **tensor**: Tensor representation of the behavioural data of type `ndarray((nsubjects, ntrials, nfeatures))`

---




### BehaviouralData.add_subject

```python
fitr.data.add_subject(self, subject_index, parameters, subject_meta)
```

Appends a new subject to the dataset

Arguments:

- **subject_index**: Integer identification for subject
- **parameters**: `list` of parameters for the subject
- **subject_meta**: Some covariates for the subject (`list`)

---




### BehaviouralData.initialize_data_dictionary

```python
fitr.data.initialize_data_dictionary(self)
```



---




### BehaviouralData.make_behavioural_ngrams

```python
fitr.data.make_behavioural_ngrams(self, n)
```

Creates N-grams of behavioural data 

---




### BehaviouralData.make_cooccurrence_matrix

```python
fitr.data.make_cooccurrence_matrix(self, k, dtype=<class 'numpy.float32'>)
```



---




### BehaviouralData.make_tensor_representations

```python
fitr.data.make_tensor_representations(self)
```

Creates a tensor with all subjects' data

#### Notes

Assumes that all subjects did same number of trials.

---




### BehaviouralData.numpy_tensor_to_bdf

```python
fitr.data.numpy_tensor_to_bdf(self, X)
```

Creates `BehaviouralData` formatted set from a dataset stored in a numpy `ndarray`.

Arguments:

- **X**: `ndarray((nsubjects, ntrials, m))` with `m` being the size of flattened single-trial data

---




### BehaviouralData.unpack_tensor

```python
fitr.data.unpack_tensor(self, x_dim, u_dim, r_dim=1, terminal_dim=1, get='sarsat')
```

Unpacks data stored in tensor format into separate arrays for states, actions, rewards, next states, and next actions.

Arguments:

x_dim : Task state space dimensionality (`int`)
u_dim : Task action space dimensionality (`int`)
r_dim : Reward dimensionality (`int`, default=1)
terminal_dim : Dimensionality of the terminal state indicator (`int` , default=1)
get : String indicating the order that data are stored in the array. Can also be shortened such that fewer elements are returned. For example, the default is `sarsat`.

Returns:

List with data, where each element is in the order of the argument `get`

---




### BehaviouralData.update

```python
fitr.data.update(self, subject_index, behav_data)
```

Adds behavioural data to the dataset

Arguments:

- **subject_index**: Integer index for the subject
- **behav_data**: 1-dimensional `ndarray` of flattened data

---



## merge_behavioural_data

```python
fitr.data.merge_behavioural_data(datalist)
```

Combines BehaviouralData objects.

Arguments:

- **datalist**: List of BehaviouralData objects

Returns:

`BehaviouralData` with data from multiple groups merged.

---


