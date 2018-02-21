import logging

import torch
import torch.nn as nn

from . import functions as L
from . import wrappers
from . import core

from ..backend import flatten


class ContextAttention(nn.Module):

    def __init__(self, input_size, output_size=1, activation='tanh', batchnorm=False, padded_input=True):
        super(ContextAttention, self).__init__()
        self.activation_name = activation
        self.padded_input = padded_input
        self.attentional_module = core.FullyConnected(input_size, input_size, activation=activation,
                                                      batchnorm=batchnorm)
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

    def __repr__(self):
        return str(self)
