import logging

import torch.nn as nn
import torch.nn.functional as F

from . import layer_utils as utils
from . import functions as L

# TODO Create abstract layers for layers with params that includes weight regularizers


def build_fully_connected(input_size, output_size, use_bias=True, activation='linear', num_layers=1, batchnorm=False,
                          input_dropout=0.0, dropout=0.0):
    layer = nn.Sequential()
    if input_dropout:
        layer.add_module(name="input-dropout", module=nn.Dropout(input_dropout))
    for i in range(num_layers):
        layer_input = input_size if i == 0 else output_size
        layer.add_module(name="fullyconnected-%s" % i, module=nn.Linear(layer_input, output_size, bias=use_bias))
        if activation != "linear":
            layer.add_module(name="{}-{}".format(activation, i), module=utils.get_activation_type(activation)())
        if batchnorm:
            layer.add_module(name="batchnorm-%s" % i, module=nn.BatchNorm1d(output_size))
        if dropout:
            layer.add_module(name="dropout-%s" % i, module=nn.Dropout(dropout))
    logging.info("Creating layer: %r" % layer)
    return layer


class FullyConnected(nn.Module):
    """Just your regular fully-connected NN layer.
        `FullyConnected` implements the operation:
        `output = activation(dot(input, kernel) + bias)`
        where `activation` is the element-wise activation function
        passed as the `activation` argument, `kernel` is a weights matrix
        created by the layer, and `bias` is a bias vector created by the layer
        (only applicable if `use_bias` is `True`).
        Note: if the input to the layer has a rank greater than 2, then
        it is flattened prior to the initial dot product with `kernel`.
        # Example
        ```python
            # A layer that takes as input tensors of shape (*, 128)
            # and outputs arrays of shape (*, 64)
            layer = FullyConnected(128, 64)
            tensor = torch.randn(32, 128)
            output = layer(tensor)
        ```
        # Arguments
            input_size: Positive integer, dimensionality of the input space.
            output_size: Positive integer, dimensionality of the input space.
            activation: String, Name of activation function to use
                (supports "tanh", "relu", and "linear").
                If you don't specify anything, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
            use_bias: Boolean, whether the layer uses a bias vector.
        # Input shape
            2D tensor with shape: `(batch_size, input_size)`.
        # Output shape
            2D tensor with shape: `(batch_size, output_size)`.
        """

    def __init__(self, input_size, output_size, use_bias=True, activation='linear', num_layers=1,
                 batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(FullyConnected, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        self.use_bias = use_bias
        self.num_layers = num_layers
        self.batchnorm = batchnorm

        # Build the layers
        self.layers = build_fully_connected(input_size, output_size, use_bias=use_bias, activation=activation,
                                            num_layers=num_layers, batchnorm=batchnorm, input_dropout=input_dropout,
                                            dropout=dropout)

    def forward(self, inputs):
        return self.layers(inputs)

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.Linear):
                logging.info("Resetting layer %s" % layer)
                layer.reset_parameters()

    def __str__(self):
        return "%r" % self.layers

    def __repr__(self):
        return str(self)


class Flatten(nn.Module):
    """Flattens the input. Does not affect the batch size.
        # Example
        ```python
            flatten = Flatten()
            tensor = torch.randn(32, 2, 3)
            # The output will be of shape (32, 6)
            output = flatten(tensor)
        ```
        """

    def __init__(self):
        super(Flatten, self).__init__()

    def __str__(self):
        return "Flatten"

    def forward(self, x):
        return L.flatten(x)

    def reset_parameters(self):
        pass

