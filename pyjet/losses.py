import torch
import torch.nn as nn
import torch.nn.functional as F

import pyjet.backend as J


def categorical_crossentropy(outputs, targets, size_average=True):
    """
    Computes the categorical crossentropy loss over some outputs and targets according the
    equation for the ith output

    -log(output[target])

    and is accumulated with a sum or average over all outputs.

    # Arguments:
        outputs -- The torch FloatTensor output from a model with the shape (N, C) where N is the
                   number of outputs and C is the number of classes.
        targets -- The torch LongTensor indicies of the ground truth with the shape (N,) where N is
                   the number of outputs and each target t is 0 <= t < C.
        size_average -- By default, the losses are averaged over observations for each minibatch.
                        However, if the field size_average is set to False, the losses are instead
                        summed for each minibatch.
    # Returns:
        A scalar tensor equal to the total loss of the output.
    """
    return F.nll_loss(outputs.clamp(J.epsilon, 1 - J.epsilon).log(), targets, size_average=size_average)


def bce_with_logits(outputs, targets, size_average=True):
    """
    Computes the binary cross entropy between targets and output's logits.

    See :class:`~torch.nn.BCEWithLogitsLoss` for details.

    # Arguments
        outputs -- A torch FloatTensor of arbitrary shape with a 1 dimensional channel axis
        targets -- A binary torch LongTensor of the same size without the channel axis
        size_average -- By default, the losses are averaged over observations for each minibatch.
                        However, if the field size_average is set to False, the losses are instead
                        summed for each minibatch.
    # Returns:
        A scalar tensor equal to the total loss of the output.

    Examples::

         >>> input = autograd.Variable(torch.randn(3), requires_grad=True)
         >>> target = autograd.Variable(torch.FloatTensor(3).random_(2))
         >>> loss = bce_with_logits(input, target)
         >>> loss.backward()
    """
    # Squeeze the output to make it 1D
    return F.binary_cross_entropy_with_logits(outputs.squeeze(1), targets.float(), size_average=size_average)
