### cast_input_to_torch


```python
cast_input_to_torch(self, x, volatile=False)
```

----

### cast_target_to_torch


```python
cast_target_to_torch(self, y, volatile=False)
```

----

### cast_output_to_numpy


```python
cast_output_to_numpy(self, preds)
```

----

### forward


```python
forward(self)
```

----

### train_on_batch


```python
train_on_batch(self, x, target, optimizer, loss_fn, metrics=())
```

----

### validate_on_batch


```python
validate_on_batch(self, x, target, metrics)
```

----

### predict_on_batch


```python
predict_on_batch(self, x)
```

----

### fit_generator


```python
fit_generator(self, generator, steps_per_epoch, epochs, optimizer, loss_fn, validation_generator=None, validation_steps=0, metrics=(), callbacks=(), initial_epoch=0)
```

----

### validate_generator


```python
validate_generator(self, val_generator, validation_steps, loss_fn=None, metrics=())
```

----

### predict_generator


```python
predict_generator(self, generator, prediction_steps, verbose=0)
```

----

### load_state


```python
load_state(self, load_path)
```

----

### save_state


```python
save_state(self, save_path)
```
