import logging

import torch
import torch.nn as nn

from . import functions as L
from . import wrappers
from . import core


class ContextAttention(nn.Module):

    def __init__(self, input_size, output_size=1, activation='tanh', batchnorm=False, padded_input=True):
        super(ContextAttention, self).__init__()
        self.activation_name = activation
        self.padded_input = padded_input
        self.hidden_layer = wrappers.TimeDistributed(core.FullyConnected(input_size, input_size, activation=activation,
                                                                         batchnorm=batchnorm))
        self.context_vector = wrappers.TimeDistributed(core.FullyConnected(input_size, output_size, use_bias=False,
                                                                           batchnorm=False))

    def forward(self, x, seq_lens=None):
        if self.padded_input:
            padded_input = x
            x = L.unpad_sequences(x, seq_lens)
        else:
            padded_input = L.pad_sequences(x)[0]
        # The input comes in as B x Li x E
        att = self.context_vector(self.hidden_layer(x))  # B x L x H
        att, _ = L.seq_softmax(att, return_padded=True)  # B x L x K
        return torch.bmm(att.transpose(1, 2), padded_input)

    def reset_parameters(self):
        self.hidden_layer.reset_paramaters()
        self.context_vector.reset_paramaters()

    def __str__(self):
        return "%r" % self.pool

    def __repr__(self):
        return str(self)
