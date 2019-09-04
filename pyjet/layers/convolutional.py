import torch.nn as nn
import torch.nn.functional as F

from . import layer
from . import layer_utils as utils

from .. import backend as J

import logging

# TODO: Add padding and cropping layers


def build_conv(
    dimensions,
    input_size,
    output_size,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    use_bias=True,
    input_activation="linear",
    activation="linear",
    num_layers=1,
    input_batchnorm=False,
    batchnorm=False,
    spectral_norm=False,
    input_dropout=0.0,
    dropout=0.0,
):
    # Create the sequential
    layer = nn.Sequential()
    # Add the input dropout
    if input_dropout:
        layer.add_module(name="input-dropout", module=nn.Dropout(input_dropout))
    if input_batchnorm:
        layer.add_module(
            name="input-batchnorm", module=Conv.bn_constructors[dimensions](input_size)
        )
    if input_activation != "linear":
        try:
            layer.add_module(
                name="input_{}".format(input_activation),
                module=utils.get_activation_type(input_activation)(inplace=True),
            )
        except TypeError:  # If inplace is not an option on the activation
            layer.add_module(
                name="input_{}".format(input_activation),
                module=utils.get_activation_type(input_activation)(),
            )
    # Add each layer
    for i in range(num_layers):
        layer_input = input_size if i == 0 else output_size
        conv = Conv.layer_constructors[dimensions](
            layer_input,
            output_size,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
        )
        if spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        layer.add_module(name="conv-%s" % i, module=conv)
        if activation != "linear":
            try:
                layer.add_module(
                    name="{}-{}".format(activation, i),
                    module=utils.get_activation_type(activation)(inplace=True),
                )
            except TypeError:  # If inplace is not an option on the activation
                layer.add_module(
                    name="{}-{}".format(activation, i),
                    module=utils.get_activation_type(activation)(),
                )
        if batchnorm:
            layer.add_module(
                name="batchnorm-%s" % i,
                module=Conv.bn_constructors[dimensions](output_size),
            )
        if dropout:
            layer.add_module(name="dropout-%s" % i, module=nn.Dropout(dropout))

    logging.info("Creating layers: %r" % layer)
    return layer


class Conv(layer.Layer):

    layer_constructors = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    bn_constructors = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}

    def __init__(
        self,
        dimensions,
        filters,
        kernel_size,
        input_shape=None,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        use_bias=True,
        input_activation="linear",
        activation="linear",
        num_layers=1,
        input_batchnorm=False,
        batchnorm=False,
        spectral_norm=False,
        input_dropout=0.0,
        dropout=0.0,
        channels_mode=J.channels_mode,
    ):
        """
        A standard convolution layer of arbitrary dimension.

        Arguments:
            dimensions {1|2|3} -- Number of dimensions to run the convolution over
            filters {int} -- Number of output filters for the convolutional layer
            kernel_size {int} -- Length of side of convolutional kernel

        Keyword Arguments:
            input_shape {tuple[dimensions+1]} -- Specify the input size to the convolutional layer (default: {None})
            stride {int} -- The stride of the kernel over the input (default: {1})
            padding {"same"|int} -- How much to pad the output with zeros by. Defaults to padding to match the input size. (default: {"same"})
            dilation {int} -- How much to dilate the convolutional kernel (default: {1})
            groups {int} -- TODO: idk what this is... (default: {1})
            use_bias {bool} -- Whether or not to use a bias value in the kernel-input dot product (default: {True})
            input_activation {str} -- Activation to use on the input. Any activation in PyTorch can be used with its lower case name. (default: {"linear"})
            activation {str} -- Activation to use on the output of the convolution. Any activation in PyTorch can be used with its lower case name. (default: {"linear"})
            num_layers {int} -- Number of convolutional layers to stack. Layers after the first will both take as input and output `output_filters` number of filters (default: {1})
            input_batchnorm {bool} -- Whether or not to run batchnorm on the input. (default: {False})
            batchnorm {bool} -- Whether or not to run batchnorm on the output of the activation. (default: {False})
            input_dropout {float} -- Dropout for the input to the layer. (default: {0.0})
            dropout {float} -- Dropout for the final output of the layer. (default: {0.0})
            channels_mode {"channels_first"|"channels_last"} -- Whether the channels (filters) dimension of the input to this layer is before or after spatial dimensions. Defaults to whatever is set in your pyjet.json  (default: {J.channels_mode})
            """

        super(Conv, self).__init__()
        # Catch any bad padding inputs (NOTE: this does not catch negative padding)
        if padding != "same" and not isinstance(padding, int):
            raise NotImplementedError("padding: %s" % padding)
        if dimensions not in [1, 2, 3]:
            raise NotImplementedError("Conv{}D".format(dimensions))

        # Set up attributes
        self.dimensions = dimensions
        self.filters = filters
        self.input_shape = input_shape
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        self.input_activation = input_activation
        self.activation = activation
        self.num_layers = num_layers
        self.input_batchnorm = input_batchnorm
        self.batchnorm = batchnorm
        self.spectral_norm = spectral_norm
        self.input_dropout = input_dropout
        self.dropout = dropout
        self.channels_mode = channels_mode

        # Build the layers
        self.conv_layers = []
        self.register_builder(self.__build_layer)

    def get_same_padding(self, input_len):
        total_padding = int(
            self.stride * (input_len - 1)
            + 1
            + self.dilation * (self.kernel_size - 1)
            - input_len
        )
        if total_padding % 2 == 1:
            pad_l = total_padding // 2
            return pad_l, total_padding - pad_l
        else:
            pad = total_padding // 2
            return pad, pad

    def get_padding(self, input_len):
        if self.padding != "same":
            return self.padding, self.padding
        else:
            return self.get_same_padding(input_len)

    def pad_input(self, x):
        raise NotImplementedError("Layer does not know how to pad input")

    def weight(self, i=0, bias=False):
        # Default case is 0 which works when there's only 1 conv layer.
        layer = getattr(self.conv_layers, f"conv-{i}")
        return layer.weight if not bias else layer.bias

    def __build_layer(self, inputs):
        if self.input_shape is None:
            self.input_shape = utils.get_input_shape(inputs)
        if self.channels_mode == "channels_last":
            input_channels = self.input_shape[-1]
        else:
            input_channels = self.input_shape[0]
        self.conv_layers = build_conv(
            self.dimensions,
            input_channels,
            self.filters,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
            use_bias=self.use_bias,
            input_activation=self.input_activation,
            activation=self.activation,
            num_layers=self.num_layers,
            input_batchnorm=self.input_batchnorm,
            batchnorm=self.batchnorm,
            spectral_norm=self.spectral_norm,
            input_dropout=self.input_dropout,
            dropout=self.dropout,
        )

    def calc_output_size(self, input_size):
        output_size = (
            input_size - self.dilation * (self.kernel_size - 1) + 2 * self.padding - 1
        ) // self.stride + 1
        return output_size

    def calc_input_size(self, output_size):
        return (
            (output_size - 1) * self.stride
            - 2 * self.padding
            + 1
            + self.dilation * (self.kernel_size - 1)
        )

    def forward(self, inputs):
        # Expect inputs as BatchSize x Length1 x ... x LengthN x Filters
        if self.channels_mode == "channels_last":
            inputs = self.fix_input(inputs)
        inputs = self.conv_layers(self.pad_input(inputs))
        if self.channels_mode == "channels_last":
            inputs = self.unfix_input(inputs)
        return inputs

    def reset_parameters(self):
        for layer in self.conv_layers:
            if any(
                isinstance(layer, self.layer_constructors[dim])
                or isinstance(layer, self.bn_constructors[dim])
                for dim in self.layer_constructors
            ):
                logging.info("Resetting layer %s" % layer)
                layer.reset_parameters()

    def __str__(self):
        return "%r" % self.conv_layers


class Conv1D(Conv):
    def __init__(
        self,
        filters,
        kernel_size,
        input_shape=None,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        use_bias=True,
        input_activation="linear",
        activation="linear",
        num_layers=1,
        input_batchnorm=False,
        batchnorm=False,
        spectral_norm=False,
        input_dropout=0.0,
        dropout=0.0,
    ):
        super(Conv1D, self).__init__(
            1,
            filters,
            kernel_size,
            input_shape=input_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            input_activation=input_activation,
            activation=activation,
            num_layers=num_layers,
            input_batchnorm=input_batchnorm,
            batchnorm=batchnorm,
            spectral_norm=spectral_norm,
            input_dropout=input_dropout,
            dropout=dropout,
        )

    def fix_input(self, inputs):
        return inputs.transpose(1, 2).contiguous()

    def unfix_input(self, outputs):
        return outputs.transpose(1, 2).contiguous()

    def pad_input(self, inputs):
        # inputs is batch_size x channels x length
        return F.pad(inputs, self.get_padding(inputs.size(2)))


class Conv2D(Conv):
    def __init__(
        self,
        filters,
        kernel_size,
        input_shape=None,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        use_bias=True,
        input_activation="linear",
        activation="linear",
        num_layers=1,
        input_batchnorm=False,
        batchnorm=False,
        spectral_norm=False,
        input_dropout=0.0,
        dropout=0.0,
    ):
        super(Conv2D, self).__init__(
            2,
            filters,
            kernel_size,
            input_shape=input_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            input_activation=input_activation,
            activation=activation,
            num_layers=num_layers,
            input_batchnorm=input_batchnorm,
            batchnorm=batchnorm,
            spectral_norm=spectral_norm,
            input_dropout=input_dropout,
            dropout=dropout,
        )

    def fix_input(self, inputs):
        return inputs.permute(0, 3, 1, 2).contiguous()

    def unfix_input(self, outputs):
        return outputs.permute(0, 2, 3, 1).contiguous()

    def pad_input(self, inputs):
        # inputs is batch_size x channels x height x width
        padding = self.get_padding(inputs.size(2)) + self.get_padding(inputs.size(3))
        return F.pad(inputs, padding)


class Conv3D(Conv):
    def __init__(
        self,
        filters,
        kernel_size,
        input_shape=None,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        use_bias=True,
        input_activation="linear",
        activation="linear",
        num_layers=1,
        input_batchnorm=False,
        batchnorm=False,
        spectral_norm=False,
        input_dropout=0.0,
        dropout=0.0,
    ):
        super(Conv3D, self).__init__(
            3,
            filters,
            kernel_size,
            input_shape=input_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            input_activation=input_activation,
            activation=activation,
            num_layers=num_layers,
            input_batchnorm=input_batchnorm,
            batchnorm=batchnorm,
            spectral_norm=spectral_norm,
            input_dropout=input_dropout,
            dropout=dropout,
        )

    def fix_input(self, inputs):
        return inputs.permute(0, 4, 1, 2, 3).contiguous()

    def unfix_input(self, outputs):
        return outputs.permute(0, 2, 3, 4, 1).contiguous()

    def pad_input(self, inputs):
        # inputs is batch_size x channels x height x width x time
        padding = (
            self.get_padding(inputs.size(2))
            + self.get_padding(inputs.size(3))
            + self.get_padding(inputs.size(4))
        )
        return F.pad(inputs, padding)
