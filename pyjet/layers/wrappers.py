import logging

import torch.nn as nn

from . import layer
from . import functions as L


class Identity(layer.Layer):
    """
    This is used to create layer wrappers without passing a layer.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# Singleton Identity layer
Identity = Identity()


class SequenceInput(layer.Layer):
    """
    Wrapper for a layer that should take in variable length sequences as inputs.
    This wrapper will take as input a list of (batch size number of) sequences.
    Before passing to its layer, the wrapper will pad the sequences to the longest
    sequence in the batch, pass to the layer, then unpad back to the list of sequence form.
    The wrapper requires that sequence lengths are not modified when passed through the layer.

    Dropout will be applied to the nonpadded sequence.
    """
    def __init__(self, wrapped_layer=Identity, input_dropout=0., dropout=0., pad_value=0.):
        super(SequenceInput, self).__init__()
        self.layer = wrapped_layer
        self.input_dropout = nn.Dropout(input_dropout)
        self.input_dropout_p = input_dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout
        self.pad_value = pad_value
        self.__descriptor = "SequenceInput(input_dropout=%s, dropout=%s, pad_value=%s)" % (
                             self.input_dropout, self.dropout, self.pad_value)
        logging.info("Wrapping layer with %s: %r" % (self.__descriptor, self.layer))

    def forward(self, x):
        if self.input_dropout_p:
            x = [self.input_dropout(sample.unsqueeze(0)).squeeze(0) for sample in x]
        x, pad_lens = L.pad_sequences(x, pad_value=self.pad_value)
        x = self.layer(x)
        x = L.unpad_sequences(x, pad_lens)
        if self.dropout_p:
            x = [self.dropout(sample.unsqueeze(0)).squeeze(0) for sample in x]
        return x

    def reset_parameters(self):
        self.layer.reset_parameters()

    def __str__(self):
        return self.__descriptor + "(%r)" % self.layer


class TimeDistributed(layer.Layer):

    def __init__(self, wrapped_layer):
        super(TimeDistributed, self).__init__()
        self.layer = wrapped_layer
        logging.info("TimeDistributing %r layer" % self.layer)

    def forward(self, x):
        x, seq_lens = L.pack_sequences(x)  # B*Li x I
        x = self.layer(x)  # B*Li x O
        x = L.unpack_sequences(x, seq_lens)
        return x

    def reset_parameters(self):
        self.layer.reset_parameters()

    def __str__(self):
        return "TimeDistributed" + "(%r)" % self.layer
