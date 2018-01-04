import torch.nn as nn
import torch.nn.functional as F

import layers.layer_utils as utils

import logging

# TODO: Add padding and cropping layers


class Conv(nn.Module):

    layer_constructors = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}

    def __init__(self, dimensions, input_size, output_size, kernel_size, stride=1, padding='same', dilation=1, groups=1,
                 bias=True, activation='linear', num_layers=1,
                 batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(Conv, self).__init__()
        # Catch any bad padding inputs (NOTE: this does not catch negative padding)
        if padding != 'same' and not isinstance(padding, int):
            raise NotImplementedError("padding: %s" % padding)
        if dimensions not in [1, 2, 3]:
            raise NotImplementedError("Conv{}D".format(dimensions))
        padding = (kernel_size - stride) // 2 if padding == 'same' else padding

        # Set up attributes
        self.dimensions = dimensions
        self.input_size = input_size
        self.output_size = output_size
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.activation_name = activation
        self.num_layers = num_layers
        self.batchnorm = batchnorm

        # Build the layers
        self.conv_layers = utils.construct_n_layers(Conv.layer_constructors[dimensions], num_layers, input_size,
                                                    output_size, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation,
                                                    groups=groups, bias=bias)
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
                             "{} Conv{}D {} x {}".format(self.activation_name, self.dimensions, self.input_size,
                                                         self.output_size),
                             "" if batchnorm else "{} Batchnorm".format(self.output_size),
                             "{} Drop".format(self.dropout_p)]
        for _ in range(num_layers):
            self.__str_params.append(
                "{} Conv{}D {} x {}".format(self.activation_name, self.dimensions, self.output_size, self.output_size))
            self.__str_params.append("" if batchnorm else "{} Batchnorm".format(self.output_size))
            self.__str_params.append("{} Drop".format(self.dropout_p))

        # Logging
        logging.info("Creating layer: {}".format(str(self)))

    def forward(self, x):
        # Expect x as BatchSize x Length x Filters. Move filters for the conv layer
        x = x.transpose(-1, 1).contiguous()
        # Run the input dropout
        if self.input_dropout_p:
            x = F.dropout(x, p=self.input_dropout_p, training=self.training)
        # Run each of the n layers
        for i, conv in enumerate(self.conv_layers):
            x = self.activation(conv(x))
            if self.batchnorm:
                x = self.bn[i](x)
            if self.dropout_p:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        # Undo it to return back to the original shape
        x = x.transpose(-1, 1).contiguous()
        return x

    def __str__(self):
        return "\n-> ".join(self.__str_params)

    def __repr__(self):
        return "-".join(self.__str_params)


class Conv1D(Conv):
    def __init__(self, input_size, output_size, kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=True,
                 activation='linear', num_layers=1,
                 batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(Conv1D, self).__init__(1,  input_size, output_size, kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, groups=groups, bias=bias,
                                     activation=activation, num_layers=num_layers,
                                     batchnorm=batchnorm,
                                     input_dropout=input_dropout, dropout=dropout)


class Conv2D(Conv):
    def __init__(self, input_size, output_size, kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=True,
                 activation='linear', num_layers=1,
                 batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(Conv2D, self).__init__(2,  input_size, output_size, kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, groups=groups, bias=bias,
                                     activation=activation, num_layers=num_layers,
                                     batchnorm=batchnorm,
                                     input_dropout=input_dropout, dropout=dropout)


class Conv3D(Conv):
    def __init__(self, input_size, output_size, kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=True,
                 activation='linear', num_layers=1,
                 batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(Conv3D, self).__init__(3,  input_size, output_size, kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, groups=groups, bias=bias,
                                     activation=activation, num_layers=num_layers,
                                     batchnorm=batchnorm,
                                     input_dropout=input_dropout, dropout=dropout)