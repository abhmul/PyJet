import logging

import torch
import torch.nn as nn

from . import layer
from . import functions as L
from . import wrappers
from . import core
from . import pooling

from ..backend import flatten


class ContextAttention(layer.Layer):

    def __init__(self, input_size, output_size=1, activation='tanh', batchnorm=False, padded_input=True, dropout=0.0):
        super(ContextAttention, self).__init__()
        self.activation_name = activation
        self.padded_input = padded_input
        self.attentional_module = core.FullyConnected(input_size, input_size, activation=activation,
                                                      batchnorm=batchnorm, dropout=dropout)
        self.context_vector = core.FullyConnected(input_size, output_size, use_bias=False, batchnorm=False)
        self.context_attention = wrappers.TimeDistributed(nn.Sequential(self.attentional_module, self.context_vector))

    def forward(self, x, seq_lens=None):
        if self.padded_input:
            padded_input = x
            x = L.unpad_sequences(x, seq_lens)
        else:
            padded_input, _ = L.pad_sequences(x)  # B x L x H
        # The input comes in as B x Li x E
        att = self.context_attention(x)  # B x L x H
        att, _ = L.seq_softmax(att, return_padded=True)  # B x L x K
        out = torch.bmm(att.transpose(1, 2), padded_input)  # B x K x H
        return out.squeeze_(1)

    def reset_parameters(self):
        self.attentional_module.reset_parameters()
        self.context_vector.reset_parameters()

    def __str__(self):
        return "%r" % self.pool


class ContextMaxPool1D(layer.Layer):

    def __init__(self, input_size, output_size=1, activation='linear', batchnorm=False, padded_input=True, dropout=0.0):
        super(ContextMaxPool1D, self).__init__()
        self.padded_input = padded_input
        self.max_pool = pooling.SequenceGlobalMaxPooling1D()
        self.context_attention = nn.ModuleList([wrappers.TimeDistributed(
            core.FullyConnected(input_size, input_size, batchnorm=batchnorm, activation=activation, dropout=dropout))
                                                for _ in range(output_size)])

    def forward(self, x, seq_lens=None):
        if self.padded_input:
            padded_input = x
            x = L.unpad_sequences(x, seq_lens)
        else:
            padded_input, _ = L.pad_sequences(x)  # B x L x H
        # The input comes in as B x Li x E
        out_heads = torch.stack([self.max_pool(head(x)) for head in self.context_attention], dim=1)  # B x K x H
        return out_heads.squeeze_(1)

    def reset_parameters(self):
        for i in range(len(self.context_attention)):
            self.context_attention[i].reset_parameters()

    def __str__(self):
        return "%r" % self.pool