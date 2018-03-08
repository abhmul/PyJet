import torch.nn as nn

from . import layer
from . import layer_utils as utils
from . import functions as L

import logging

# TODO: Add padding and cropping layers


def build_conv(dimensions, input_size, output_size, kernel_size, stride=1, padding=0, dilation=1, groups=1,
               use_bias=True, input_activation='linear', activation='linear', num_layers=1,
               input_batchnorm=False, batchnorm=False,
               input_dropout=0.0, dropout=0.0):
    # Create the sequential
    layer = nn.Sequential()
    # Add the input dropout
    if input_dropout:
        layer.add_module(name="input-dropout", module=nn.Dropout(input_dropout))
    if input_batchnorm:
        layer.add_module(name="input-batchnorm", module=Conv.bn_constructors[dimensions](input_size))
    if input_activation != 'linear':
        layer.add_module(name="input_{}".format(input_activation), module=utils.get_activation_type(input_activation)())
    # Add each layer
    for i in range(num_layers):
        layer_input = input_size if i == 0 else output_size
        layer.add_module(name="conv-%s" % i,
                         module=Conv.layer_constructors[dimensions](layer_input, output_size, kernel_size,
                                                                    stride=stride, padding=padding, dilation=dilation,
                                                                    groups=groups, bias=use_bias))
        if activation != "linear":
            layer.add_module(name="{}-{}".format(activation, i), module=utils.get_activation_type(activation)())
        if batchnorm:
            layer.add_module(name="batchnorm-%s" % i, module=Conv.bn_constructors[dimensions](output_size))
        if dropout:
            layer.add_module(name="dropout-%s" % i, module=nn.Dropout(dropout))
    return layer


class Conv(layer.Layer):

    layer_constructors = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    bn_constructors = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}

    def __init__(self, dimensions, input_size, output_size, kernel_size, stride=1, padding='same', dilation=1, groups=1,
                 use_bias=True, input_activation='linear', activation='linear', num_layers=1,
                 input_batchnorm=False, batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(Conv, self).__init__()
        # Catch any bad padding inputs (NOTE: this does not catch negative padding)
        if padding != 'same' and not isinstance(padding, int):
            raise NotImplementedError("padding: %s" % padding)
        if dimensions not in [1, 2, 3]:
            raise NotImplementedError("Conv{}D".format(dimensions))
        padding = (kernel_size - 1) // 2 if padding == 'same' else padding

        # Set up attributes
        self.dimensions = dimensions
        self.input_size = input_size
        self.output_size = output_size
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        self.input_activation_name = input_activation
        self.activation_name = activation
        self.num_layers = num_layers
        self.input_batchnorm = input_batchnorm
        self.batchnorm = batchnorm

        # Build the layers
        self.conv_layers = build_conv(dimensions, input_size, output_size, kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups,
                                      use_bias=use_bias, input_activation=input_activation, activation=activation,
                                      num_layers=num_layers, input_batchnorm=input_batchnorm, batchnorm=batchnorm,
                                      input_dropout=input_dropout, dropout=dropout)
        logging.info("Creating layers: %r" % self.conv_layers)

    def calc_output_size(self, input_size):
        """
        NOTE: This is designed for pytorch longtensors, if you pass an integer, make sure to cast it back to an
        integer as python3 will perform float division on it
        """
        output_size = (input_size - self.dilation * (self.kernel_size - 1) + 2 * self.padding - 1) / self.stride + 1
        return output_size

    def calc_input_size(self, output_size):
        return (output_size - 1) * self.stride - 2 * self.padding + 1 + self.dilation * (self.kernel_size - 1)

    def forward(self, inputs):
        # Expect inputs as BatchSize x Length1 x ... x LengthN x Filters
        return self.unfix_input(self.conv_layers(self.fix_input(inputs)))

    def reset_parameters(self):
        for layer in self.conv_layers:
            if any(isinstance(layer, self.layer_constructors[dim]) or isinstance(layer, self.bn_constructors[dim])
                   for dim in self.layer_constructors):
                logging.info("Resetting layer %s" % layer)
                layer.reset_parameters()

    def __str__(self):
        return "%r" % self.conv_layers


class Conv1D(Conv):
    def __init__(self, input_size, output_size, kernel_size, stride=1, padding='same', dilation=1, groups=1,
                 use_bias=True, input_activation='linear', activation='linear', num_layers=1,
                 input_batchnorm=False, batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(Conv1D, self).__init__(1, input_size, output_size, kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, groups=groups, use_bias=use_bias,
                                     input_activation=input_activation, activation=activation, num_layers=num_layers,
                                     input_batchnorm=input_batchnorm, batchnorm=batchnorm,
                                     input_dropout=input_dropout, dropout=dropout)

    def fix_input(self, inputs):
        return inputs.transpose(1, 2).contiguous()

    def unfix_input(self, outputs):
        return outputs.transpose(1, 2).contiguous()


class SequenceConv1D(Conv1D):
    def __init__(self, input_size, output_size, kernel_size,
                 use_bias=True, input_activation='linear', activation='linear',
                 input_batchnorm=False, batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(SequenceConv1D, self).__init__(input_size, output_size, kernel_size, stride=1, padding='same',
                                     dilation=1, groups=1, use_bias=use_bias,
                                     input_activation='linear', activation='linear', num_layers=1,
                                     input_batchnorm=False, batchnorm=False,
                                     input_dropout=False, dropout=False)
        identity = lambda x: x
        self.input_dropout = identity if input_dropout == 0. else nn.Dropout(input_dropout)
        self.dropout = identity if dropout == 0. else nn.Dropout(dropout)
        self.input_activation = identity if input_activation == 'linear' else utils.get_activation_type(input_activation)()
        self.activation = identity if activation == 'linear' else utils.get_activation_type(activation)()
        self.input_batchnorm = None if not input_batchnorm else nn.BatchNorm1d(input_size)
        self.batchnorm = None if not batchnorm else nn.BatchNorm1d(output_size)

    def forward(self, seq_inputs):
        seq_inputs = [self.input_dropout(sample.unsqueeze(0)).squeeze(0) for sample in seq_inputs]
        if self.input_batchnorm is not None:
            padded_inputs, seq_lens = L.pad_sequences(seq_inputs)
            padded_inputs = self.unfix_input(self.input_batchnorm(self.fix_input(padded_inputs)))
            seq_inputs = L.unpad_sequences(padded_inputs, seq_lens)
        seq_inputs = [self.activation(super(SequenceConv1D, self).forward(self.input_activation(sample.unsqueeze(0)))).squeeze(0) for sample in seq_inputs]
        if self.batchnorm is not None:
            padded_inputs, seq_lens = L.pad_sequences(seq_inputs)
            padded_inputs = self.unfix_input(self.batchnorm(self.fix_input(padded_inputs)))
            seq_inputs = L.unpad_sequences(padded_inputs, seq_lens)
        seq_inputs = [self.dropout(sample.unsqueeze(0)).squeeze(0) for sample in seq_inputs]
        return seq_inputs

    def reset_parameters(self):
        if self.batchnorm is not None:
            self.batchnorm.reset_parameters()
        if self.input_batchnorm is not None:
            self.input_batchnorm.reset_parameters()
        super().reset_parameters()


class Conv2D(Conv):
    def __init__(self, input_size, output_size, kernel_size, stride=1, padding='same', dilation=1, groups=1,
                 use_bias=True, input_activation='linear', activation='linear', num_layers=1,
                 input_batchnorm=False, batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(Conv2D, self).__init__(2, input_size, output_size, kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, groups=groups, use_bias=use_bias,
                                     input_activation=input_activation, activation=activation, num_layers=num_layers,
                                     input_batchnorm=input_batchnorm, batchnorm=batchnorm,
                                     input_dropout=input_dropout, dropout=dropout)

    def fix_input(self, inputs):
        return inputs.permute(0, 3, 1, 2).contiguous()

    def unfix_input(self, outputs):
        return outputs.permute(0, 2, 3, 1).contiguous()


class Conv3D(Conv):
    def __init__(self, input_size, output_size, kernel_size, stride=1, padding='same', dilation=1, groups=1,
                 use_bias=True, input_activation='linear', activation='linear', num_layers=1,
                 input_batchnorm=False, batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(Conv3D, self).__init__(3, input_size, output_size, kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, groups=groups, use_bias=use_bias,
                                     input_activation=input_activation, activation=activation, num_layers=num_layers,
                                     input_batchnorm=input_batchnorm, batchnorm=batchnorm,
                                     input_dropout=input_dropout, dropout=dropout)
