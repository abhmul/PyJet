import logging

import torch
import torch.nn as nn

from . import layer
from . import layer_utils as utils
from . import functions as L
from . import wrappers
from . import core
from . import pooling

from .. import backend as J


class ContextAttention(layer.Layer):
    def __init__(
        self,
        units,
        input_shape=None,
        activation="tanh",
        batchnorm=False,
        padded_input=True,
        dropout=0.0,
    ):
        super(ContextAttention, self).__init__()
        self.units = units
        self.input_shape = input_shape
        self.activation_name = activation
        self.batchnorm = batchnorm
        self.padded_input = padded_input
        self.dropout = dropout

        self.attentional_module = None
        self.context_vector = None
        self.context_attention = None

        # Registrations
        self.register_builder(self.__build_layer)

    def __build_layer(self, inputs):
        assert self.input_shape is None
        # Use the 0th input since the inputs are time distributed
        self.input_shape = utils.get_input_shape(inputs[0])
        self.attentional_module = core.FullyConnected(
            self.input_shape[0],
            input_shape=self.input_shape,
            activation=self.activation_name,
            batchnorm=self.batchnorm,
            dropout=self.dropout,
        )
        self.context_vector = core.FullyConnected(
            self.units, input_shape=self.input_shape, use_bias=False, batchnorm=False
        )
        self.context_attention = wrappers.TimeDistributed(
            nn.Sequential(self.attentional_module, self.context_vector)
        )

    def forward(self, x, seq_lens=None):
        if self.padded_input:
            padded_input = x
            if seq_lens is None:
                seq_lens = J.LongTensor([x.size(1)] * x.size(0))
            x = L.unpad_sequences(x, seq_lens)
        else:
            padded_input, _ = L.pad_sequences(x)  # B x L x H
        # Build the layer if we don't know the input shape
        if not self.built:
            self.__build_layer(x)

        # The input comes in as B x Li x E
        att = self.context_attention(x)  # B x L x H
        att, _ = L.seq_softmax(att, return_padded=True)  # B x L x K
        out = torch.bmm(att.transpose(1, 2), padded_input)  # B x K x H
        return out.squeeze_(1)

    def reset_parameters(self):
        if self.built:
            self.attentional_module.reset_parameters()
            self.context_vector.reset_parameters()

    def __str__(self):
        return "%r" % self.context_attention


class ContextMaxPool1D(layer.Layer):
    def __init__(
        self,
        units=1,
        input_shape=None,
        activation="linear",
        batchnorm=False,
        padded_input=True,
        dropout=0.0,
    ):
        super(ContextMaxPool1D, self).__init__()
        self.units = units
        self.input_shape = input_shape
        self.activation = activation
        self.batchnorm = batchnorm
        self.padded_input = padded_input
        self.dropout = dropout

        self.max_pool = pooling.SequenceGlobalMaxPooling1D()
        self.context_attention = None

        # Registrations
        self.register_builder(self.__build_layer)

    def __build_layer(self, inputs):
        assert self.input_shape is None
        # Use the 0th input since the inputs are time distributed
        self.input_shape = utils.get_input_shape(inputs[0])
        self.context_attention = nn.ModuleList(
            [
                wrappers.TimeDistributed(
                    core.FullyConnected(
                        self.input_shape[0],
                        input_shape=self.input_shape,
                        batchnorm=self.batchnorm,
                        activation=self.activation,
                        dropout=self.dropout,
                    )
                    for _ in range(self.units)
                )
            ]
        )

    def forward(self, x, seq_lens=None):
        if self.padded_input:
            padded_input = x
            x = L.unpad_sequences(x, seq_lens)
        else:
            padded_input, _ = L.pad_sequences(x)  # B x L x H

        # Build the layer if we don't know the input shape
        if not self.built:
            self.__build_layer(x)
        # The input comes in as B x Li x E
        out_heads = torch.stack(
            [self.max_pool(head(x)) for head in self.context_attention], dim=1
        )  # B x K x H
        return out_heads.squeeze_(1)

    def reset_parameters(self):
        if self.built:
            for i in range(len(self.context_attention)):
                self.context_attention[i].reset_parameters()

    def __str__(self):
        return "%r" % self.context_attention
