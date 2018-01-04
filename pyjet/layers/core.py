import logging

import torch.nn as nn
import torch.nn.functional as F

import layers.layer_utils as utils
import layers.functions as L

# TODO Create abstract layers for layers with params that includes weight regularizers


class FullyConnected(nn.Module):

    def __init__(self, input_size, output_size, bias=True, activation='linear', num_layers=1,
                 batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(FullyConnected, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        self.bias = bias
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
            input_size, output_size, "bias" if bias else "no bias"))
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

    def __init__(self):
        super(Flatten, self).__init__()

    def __str__(self):
        return "Flatten"

    def forward(self, x):
        return L.flatten(x)

