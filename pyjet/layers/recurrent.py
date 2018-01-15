import logging
from functools import partialmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rnn(rnn_type, input_size, output_size, num_layers=1, bidirectional=False,
              input_dropout=0.0, dropout=0.0):
    # Create the sequential
    layer = nn.Sequential()
    # Add the input dropout
    if input_dropout:
        layer.add_module(name="input-dropout", module=nn.Dropout(input_dropout))
    layer.add_module(name="rnn", module=RNN.layer_constructors[rnn_type](input_size, output_size, num_layers=num_layers, dropout=dropout,
                                                      bidirectional=bidirectional, batch_first=True))
    logging.info("Creating layer: %r" % layer)
    return layer


class RNN(nn.Module):

    layer_constructors = {'gru': nn.GRU, 'lstm': nn.LSTM, "tanh_simple": partialmethod(nn.RNN, nonlinearity='tanh'),
                          "relu_simple": partialmethod(nn.RNN, nonlinearity='relu')}

    def __init__(self, rnn_type, input_size, output_size, num_layers=1, bidirectional=False,
                 input_dropout=0.0, dropout=0.0, return_sequences=False, return_state=False):
        super(RNN, self).__init__()
        output_size = output_size // 2 if bidirectional else output_size

        # Set up the attributes
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.return_sequences = return_sequences
        self.return_state = return_state

        # Build the layers
        self.rnn_layers = build_rnn(rnn_type, input_size, output_size, num_layers=num_layers,
                                    bidirectional=bidirectional, input_dropout=input_dropout, dropout=dropout)

    def forward(self, x):
        x, states = self.rnn_layers(x)
        if not self.return_sequences:
            x = x[:, -1]
        if self.return_state:
            return x, states
        return x

    def __str__(self):
        return ("%r\n\treturn_sequences={}, return_state={}" % self.rnn_layers).format(self.return_sequences,
                                                                                       self.return_state)

    def __repr__(self):
        return str(self)


class SimpleRNN(RNN):
    def __init__(self, input_size, output_size, num_layers=1, bidirectional=False,
                 input_dropout=0.0, dropout=0.0, nonlinearity='tanh', return_sequences=False, return_state=False):
        rnn_type = nonlinearity + "_" + "simple"
        super(SimpleRNN, self).__init__(rnn_type, input_size, output_size, num_layers=num_layers, bidirectional=bidirectional,
                                        input_dropout=input_dropout, dropout=dropout, return_sequences=return_sequences,
                                        return_state=return_state)


class GRU(RNN):
    def __init__(self, input_size, output_size, num_layers=1, bidirectional=False,
                 input_dropout=0.0, dropout=0.0, return_sequences=False, return_state=False):
        super(GRU, self).__init__('gru', input_size, output_size, num_layers=num_layers, bidirectional=bidirectional,
                                  input_dropout=input_dropout, dropout=dropout, return_sequences=return_sequences,
                                  return_state=return_state)


class LSTM(RNN):
    def __init__(self, input_size, output_size, num_layers=1, bidirectional=False,
                 input_dropout=0.0, dropout=0.0, return_sequences=False, return_state=False):
        super(LSTM, self).__init__('lstm', input_size, output_size, num_layers=num_layers, bidirectional=bidirectional,
                                   input_dropout=input_dropout, dropout=dropout, return_sequences=return_sequences,
                                   return_state=return_state)