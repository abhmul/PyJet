import logging

import torch
import torch.nn as nn

from . import layer


class Concatenate(layer.Layer):
    """Layer that concatenates a list of inputs.
    It takes as input a list of tensors,
    all of the same shape except for the concatenation dim, the
    dimension over which to concatenate,
    and returns a single tensor, the concatenation of all inputs.
    """
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, seq, dim=-1):
        if dim >= 0:
            dim += 1
        return torch.cat(seq, dim=dim)


class Add(layer.Layer):
    """Layer that adds a list of inputs.
    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    # Examples
    ```python
        import pyjet
        input1 = torch.randn(32, 16)
        x1 = pyjet.layers.FullyConnected(8, activation='relu')(input1)
        input2 = torch.randn(32, 16)
        x2= pyjet.layers.FullyConnected(8, activation='relu')(input2)
        added = pyjet.layers.Add()([x1, x2])  # equivalent to added = x1 + x2
        out = pyjet.layers.FullyConnected(4)(added)
        ```
    """
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, seq):
        return sum(seq)
