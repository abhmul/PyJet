import torch.nn as nn

from . import layer_utils as utils

import logging

# TODO: Add padding and cropping layers


def build_conv(dimensions, input_size, output_size, kernel_size, stride=1, padding=0, dilation=1, groups=1,
               use_bias=True, activation='linear', num_layers=1,
               batchnorm=False,
               input_dropout=0.0, dropout=0.0):
    # Create the sequential
    layer = nn.Sequential()
    # Add the input dropout
    if input_dropout:
        layer.add_module(name="input-dropout", module=nn.Dropout(input_dropout))
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


class Conv(nn.Module):

    layer_constructors = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    bn_constructors = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}

    def __init__(self, dimensions, input_size, output_size, kernel_size, stride=1, padding='same', dilation=1, groups=1,
                 use_bias=True, activation='linear', num_layers=1,
                 batchnorm=False,
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
        self.activation_name = activation
        self.num_layers = num_layers
        self.batchnorm = batchnorm

        # Build the layers
        self.conv_layers = build_conv(dimensions, input_size, output_size, kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups,
                                      use_bias=use_bias, activation=activation, num_layers=num_layers,
                                      batchnorm=batchnorm, input_dropout=input_dropout, dropout=dropout)
        logging.info("Creating layers: %r" % self.conv_layers)

    def calc_output_size(self, input_size):
        output_size = (input_size - self.dilation * (self.kernel_size - 1) + 2 * self.padding - 1) // self.stride + 1
        return output_size

    def forward(self, inputs):
        # Expect inputs as BatchSize x Filters x Length1 x ... x LengthN
        return self.conv_layers(inputs)

    def __str__(self):
        return "%r" % self.conv_layers

    def __repr__(self):
        return str(self)


class Conv1D(Conv):
    def __init__(self, input_size, output_size, kernel_size, stride=1, padding='same', dilation=1, groups=1,
                 use_bias=True, activation='linear', num_layers=1,
                 batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(Conv1D, self).__init__(1, input_size, output_size, kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, groups=groups, use_bias=use_bias,
                                     activation=activation, num_layers=num_layers,
                                     batchnorm=batchnorm,
                                     input_dropout=input_dropout, dropout=dropout)

    def fix_input(self, inputs):
        return inputs.transpose(1, 2)

    def unfix_input(self, outputs):
        return outputs.transpose(1, 2)


class Conv2D(Conv):
    def __init__(self, input_size, output_size, kernel_size, stride=1, padding='same', dilation=1, groups=1,
                 use_bias=True, activation='linear', num_layers=1,
                 batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(Conv2D, self).__init__(2, input_size, output_size, kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, groups=groups, use_bias=use_bias,
                                     activation=activation, num_layers=num_layers,
                                     batchnorm=batchnorm,
                                     input_dropout=input_dropout, dropout=dropout)


class Conv3D(Conv):
    def __init__(self, input_size, output_size, kernel_size, stride=1, padding='same', dilation=1, groups=1,
                 use_bias=True, activation='linear', num_layers=1,
                 batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(Conv3D, self).__init__(3, input_size, output_size, kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, groups=groups, use_bias=use_bias,
                                     activation=activation, num_layers=num_layers,
                                     batchnorm=batchnorm,
                                     input_dropout=input_dropout, dropout=dropout)
