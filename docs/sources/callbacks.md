<span style="float:right;">[[source]](https://github.com/PyJet/tree/master/pyjet/pyjet/callbacks.py#L153)</span>
### Plotter

```python
pyjet.callbacks.Plotter(monitor, scale='linear', plot_during_train=True, save_to_file=None, block_on_end=True)
```

----

<span style="float:right;">[[source]](https://github.com/PyJet/tree/master/pyjet/pyjet/callbacks.py#L201)</span>
### MetricLogger

```python
pyjet.callbacks.MetricLogger(log_fname)
```

----

<span style="float:right;">[[source]](https://github.com/PyJet/tree/master/pyjet/pyjet/callbacks.py#L224)</span>
### ReduceLROnPlateau

```python
pyjet.callbacks.ReduceLROnPlateau(optimizer, monitor, monitor_val=True, factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
```

Reduce learning rate when a metric has stopped improving.
Models often benefit from reducing the learning rate by a factor
of 2-10 once learning stagnates. This callback monitors a
quantity and if no improvement is seen for a 'patience' number
of epochs, the learning rate is reduced.
__Example__

```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.fit(X_train, Y_train, callbacks=[reduce_lr])
```
__Arguments__

- __optimizer__: the pytorch optimizer to modify
- __monitor__: quantity to be monitored.
- __monitor_val__: whether or not to monitor the validation quantity.
- __factor__: factor by which the learning rate will
be reduced. new_lr = lr * factor
- __patience__: number of epochs with no improvement
after which learning rate will be reduced.
- __verbose__: int. 0: quiet, 1: update messages.
- __mode__: one of {auto, min, max}. In `min` mode,
lr will be reduced when the quantity
monitored has stopped decreasing; in `max`
mode it will be reduced when the quantity
monitored has stopped increasing; in `auto`
mode, the direction is automatically inferred
from the name of the monitored quantity.
- __epsilon__: threshold for measuring the new optimum,
to only focus on significant changes.
- __cooldown__: number of epochs to wait before resuming
normal operation after lr has been reduced.
- __min_lr__: lower bound on the learning rate.

----

<span style="float:right;">[[source]](https://github.com/PyJet/tree/master/pyjet/pyjet/callbacks.py#L344)</span>
### LRScheduler

```python
pyjet.callbacks.LRScheduler(optimizer, schedule, verbose=0)
```

----

<span style="float:right;">[[source]](https://github.com/PyJet/tree/master/pyjet/pyjet/callbacks.py#L9)</span>
### Callback

```python
pyjet.callbacks.Callback()
```

Abstract base class used to build new callbacks.
__Properties__

- __params__: dict. Training parameters
(eg. verbosity, batch size, number of epochs...).
- __model__: instance of `keras.models.Model`.
Reference of the model being trained.
The `logs` dictionary that callback methods
take as argument will contain keys for quantities relevant to
the current batch or epoch.
Currently, the `.fit()` method of the `Sequential` model class
will include the following quantities in the `logs` that
it passes to its callbacks:
- __on_epoch_end__: logs include `acc` and `loss`, and
optionally include `val_loss`
(if validation is enabled in `fit`), and `val_acc`
(if validation and accuracy monitoring are enabled).
- __on_batch_begin__: logs include `size`,
the number of samples in the current batch.
- __on_batch_end__: logs include `loss`, and optionally `acc`
(if accuracy monitoring is enabled).

----

<span style="float:right;">[[source]](https://github.com/PyJet/tree/master/pyjet/pyjet/callbacks.py#L60)</span>
### ModelCheckpoint

```python
pyjet.callbacks.ModelCheckpoint(filepath, monitor, monitor_val=True, verbose=0, save_best_only=False, mode='auto', period=1)
```

Save the model after every epoch.
`filepath` can contain named formatting options,
which will be filled the value of `epoch` and
keys in `logs` (passed in `on_epoch_end`).
For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
then the model checkpoints will be saved with the epoch number and
the validation loss in the filename.
__Arguments__

- __filepath__: string, path to save the model file.
- __monitor__: quantity to monitor.
- __monitor_val__: whether or not to monitor the validation quantity.
- __verbose__: verbosity mode, 0 or 1.
- __save_best_only__: if `save_best_only=True`,
the latest best model according to
the quantity monitored will not be overwritten.
- __mode__: one of {auto, min, max}.
If `save_best_only=True`, the decision
to overwrite the current save file is made
based on either the maximization or the
minimization of the monitored quantity. For `val_acc`,
this should be `max`, for `val_loss` this should
be `min`, etc. In `auto` mode, the direction is
automatically inferred from the name of the monitored quantity.
- __save_weights_only__: if True, then only the model's weights will be
saved (`model.save_weights(filepath)`), else the full model
is saved (`model.save(filepath)`).
- __period__: Interval (number of epochs) between checkpoints.
