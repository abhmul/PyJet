<span style="float:right;">[[source]](https://github.com/PyJet/tree/master/pyjet/pyjet/layers/core.py#L30)</span>
### FullyConnected

```python
pyjet.layers.FullyConnected(input_size, output_size, use_bias=True, activation='linear', num_layers=1, batchnorm=False, input_dropout=0.0, dropout=0.0)
```

Just your regular fully-connected NN layer.
`FullyConnected` implements the operation:
`output = activation(dot(input, kernel) + bias)`
where `activation` is the element-wise activation function
passed as the `activation` argument, `kernel` is a weights matrix
created by the layer, and `bias` is a bias vector created by the layer
(only applicable if `use_bias` is `True`).
- __Note__: if the input to the layer has a rank greater than 2, then
it is flattened prior to the initial dot product with `kernel`.
__Example__

```python
# A layer that takes as input tensors of shape (*, 128)
# and outputs arrays of shape (*, 64)
layer = FullyConnected(128, 64)
tensor = torch.randn(32, 128)
output = layer(tensor)
```
__Arguments__

- __input_size__: Positive integer, dimensionality of the input space.
- __output_size__: Positive integer, dimensionality of the input space.
- __activation__: String, Name of activation function to use
(supports "tanh", "relu", and "linear").
If you don't specify anything, no activation is applied
(ie. "linear" activation: `a(x) = x`).
- __use_bias__: Boolean, whether the layer uses a bias vector.
__Input shape__

2D tensor with shape: `(batch_size, input_size)`.
__Output shape__

2D tensor with shape: `(batch_size, output_size)`.

----

<span style="float:right;">[[source]](https://github.com/PyJet/tree/master/pyjet/pyjet/layers/core.py#L88)</span>
### Flatten

```python
pyjet.layers.Flatten()
```

Flattens the input. Does not affect the batch size.
__Example__

```python
flatten = Flatten()
tensor = torch.randn(32, 2, 3)
# The output will be of shape (32, 6)
output = flatten(tensor)
```
