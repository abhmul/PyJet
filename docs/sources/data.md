<span style="float:right;">[[source]](https://github.com/PyJet/tree/master/pyjet/pyjet/data.py#L11)</span>
### Dataset

```python
pyjet.data.Dataset()
```


An abstract container for data designed to be passed to a model.
This container should implement create_batch. It is only necessary
to implement validation_split() if you use this module to split your
data into a train and test set. Same goes for kfold()

- ____Note__:__

Though not forced, a Dataset is really a constant object. Once created,
it should not be mutated in any way.

----

<span style="float:right;">[[source]](https://github.com/PyJet/tree/master/pyjet/pyjet/data.py#L184)</span>
### NpDataset

```python
pyjet.data.NpDataset(x, y=None)
```


A Dataset that is built from numpy data.

__Arguments__

x -- The input data as a numpy array
y -- The target data as a numpy array (optional)

----

<span style="float:right;">[[source]](https://github.com/PyJet/tree/master/pyjet/pyjet/data.py#L298)</span>
### HDF5Dataset

```python
pyjet.data.HDF5Dataset(x, y=None)
```

----

<span style="float:right;">[[source]](https://github.com/PyJet/tree/master/pyjet/pyjet/data.py#L305)</span>
### TorchDataset

```python
pyjet.data.TorchDataset(x, y=None)
```

----

<span style="float:right;">[[source]](https://github.com/PyJet/tree/master/pyjet/pyjet/data.py#L72)</span>
### DatasetGenerator

```python
pyjet.data.DatasetGenerator(dataset, steps_per_epoch=None, batch_size=None, shuffle=True, seed=None)
```


An iterator to create batches for a model using a Dataset. 2 of the
following must be defined
-- The input Dataset's length
-- steps_per_epoch
-- batch_size
Also, if the Dataset's length is not defined, its create_batch method
should not take any inputs

__Arguments__

dataset -- the dataset to generate from
steps_per_epoch -- The number of iterations in one epoch (optional)
batch_size -- The number of samples in one batch (optional)
shuffle -- Whether or not to shuffle the dataset before each epoch
- __default__: True
seed -- A seed for the random number generator (optional).
