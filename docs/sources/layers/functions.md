### pad_tensor


```python
pyjet.layers.pad_tensor(tensor, length, pad_value=0.0, dim=0)
```

----

### pad_sequences


```python
pyjet.layers.pad_sequences(tensors, pad_value=0.0, length_last=False)
```

----

### unpad_sequences


```python
pyjet.layers.unpad_sequences(padded_tensors, seq_lens, length_last=False)
```

----

### pack_sequences


```python
pyjet.layers.pack_sequences(tensors)
```

----

### unpack_sequences


```python
pyjet.layers.unpack_sequences(packed_tensors, seq_lens)
```

----

### kmax_pooling


```python
pyjet.layers.kmax_pooling(x, dim, k)
```

----

### pad_numpy_to_length


```python
pyjet.layers.pad_numpy_to_length(x, length)
```

----

### seq_softmax


```python
pyjet.layers.seq_softmax(x, return_padded=False)
```
