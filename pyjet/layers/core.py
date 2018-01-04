import logging

import torch.nn as nn
import torch.nn.functional as F

import layers.layer_utils as utils
import layers.functions as L

# TODO Create abstract layers for layers with params that includes weight regularizers


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
        self.linear_layers = utils.construct_n_layers(nn.Linear, num_layers, input_size, output_size, bias=True)
        # Add the extra stuff
        self.activation = utils.get_activation_type(activation)
        if batchnorm:
            self.bn = nn.ModuleList([nn.BatchNorm1d(output_size) for _ in range(num_layers)])
        else:
            self.bn = []
        self.input_dropout_p = input_dropout
        self.dropout_p = dropout

        # Used for constructing string representation
        self.__str_params = ["{} Drop".format(self.input_dropout_p),
                             "{} FullyConnected {} x {}".format(self.activation_name, self.input_size, self.output_size),
                             "" if batchnorm else "{} Batchnorm".format(self.output_size),
                             "{} Drop".format(self.dropout_p)]
        for _ in range(num_layers):
            self.__str_params.append("{} FullyConnected {} x {}".format(self.activation_name, self.output_size, self.output_size))
            self.__str_params.append("" if batchnorm else "{} Batchnorm".format(self.output_size))
            self.__str_params.append("{} Drop".format(self.dropout_p))
        # Logging
        logging.info("Using Linear layer with {} input, {} output, and {}".format(
            input_size, output_size, "bias" if use_bias else "no bias"))
        logging.info("Using activation %s" % self.activation.__name__)
        logging.info("Using batchnorm1d" if self.bn is not None else "Not using batchnorm1d")
        logging.info("Using {} input dropout and {} dropout".format(self.input_dropout_p, self.dropout_p))

    def forward(self, x):
        # Run the input dropout
        if self.input_dropout_p:
            x = F.dropout(x, p=self.input_dropout_p, training=self.training)
        # Run each of the n layers
        for i, linear in enumerate(self.linear_layers):
            x = self.activation(linear(x))
            if self.batchnorm:
                x = self.bn[i](x.unsqueeze(-1)).squeeze(-1)
            if self.dropout_p:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x

    def __str__(self):
        return "\n-> ".join(self.__str_params)

    def __repr__(self):
        return "-".join(self.__str_params)


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

