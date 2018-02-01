import logging
from functools import partialmethod

import torch
import torch.nn as nn
from torch.autograd import Variable

from .. import backend as J
from . import functions as L


class Identity(nn.Module):
    """
    This is used to create layer wrappers without passing a layer.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    def __repr__(self):
        return "Identity()"

    def __str__(self):
        return repr(self)


# Singleton Identity layer
Identity = Identity()


class SequenceInput(nn.Module):
    """
    Wrapper for a layer that should take in variable length sequences as inputs.
    This wrapper will take as input a list of (batch size number of) sequences.
    Before passing to its layer, the wrapper will pad the sequences to the longest
    sequence in the batch, pass to the layer, then unpad back to the list of sequence form.
    The wrapper requires that sequence lengths are not modified when passed through the layer.

    Dropout will be applied to the nonpadded sequence.
    """
    def __init__(self, layer=Identity, input_dropout=0., dropout=0., pad_value=0.):
        super(SequenceInput, self).__init__()
        self.layer = layer
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

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return self.__descriptor + "(%r)" % self.layer


# class MaskedInput(nn.Module):
#     """
#     Wrapper for a layer that takes in sequences of variable length as inputs that have
#     been padded. This wrapper will take as input a padded torch tensor where the sequence
#     length varies along the first dimension of each sample as well as a list of lengths of
#     each sequence in the batch. The layer will pass the input through the layer then mask
#     the padded regions of the output of the layer to cut the gradient.
#     The wrapper requires that sequence lengths are not modified when passed through the layer.
#     """
#
#     def __init__(self, layer=Identity, mask_value=0.):
#         super(MaskedInput, self).__init__()
#         self.layer = layer
#         if mask_value == 'min':
#             self.mask_value_factory = torch.min
#         else:
#             self.mask_value_factory = lambda x: Variable(mask_value)
#         self.mask_value = mask_value
#         self.masker = None
#         self.mask_valuer = None
#         self.__descriptor = "MaskedInput(mask_value=%s)" % self.mask_value
#         logging.info("Wrapping layer with %s: %r" % (self.__descriptor, self.layer))
#
#     def forward(self, x, seq_lens):
#         if self.masker is None or x.size() != self.masker.size():
#             self.masker = J.zeros(*x.size()).byte()
#             self.mask_valuerz = J.zeros(*x.size()).byte()
#         x = self.layer(x)
#         # Create the mask
#         return xVariable(self.masker), self.mask_value_factory(x).data[0])
#
#     def __str__(self):
#         return repr(self)
#
#     def __repr__(self):
#         return self.__descriptor + "(%r)" % self.layer
