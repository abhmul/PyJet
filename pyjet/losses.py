import torch
import torch.nn as nn
import torch.nn.functional as F

import pyjet.backend as J


def categorical_crossentropy(outputs, targets, reduction="elementwise_mean"):
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
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will be applied,
            'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'elementwise_mean'
    # Returns:
        A scalar tensor equal to the total loss of the output.
    """
    return F.nll_loss(outputs.clamp(J.epsilon, 1 - J.epsilon).log(), targets, reduction="elementwise_mean")


def bce_with_logits(outputs, targets, reduction="elementwise_mean"):
    """
    Computes the binary cross entropy between targets and output's logits.

    See :class:`~torch.nn.BCEWithLogitsLoss` for details.

    # Arguments
        outputs -- A torch FloatTensor of arbitrary shape with a 1 dimensional channel axis
        targets -- A binary torch LongTensor of the same size without the channel axis
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will be applied,
            'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'elementwise_mean'
    # Returns:
        A scalar tensor equal to the total loss of the output.

    Examples::

         >>> input = autograd.Variable(torch.randn(3), requires_grad=True)
         >>> target = autograd.Variable(torch.FloatTensor(3).random_(2))
         >>> loss = bce_with_logits(input, target)
         >>> loss.backward()
    """
    # Squeeze the output to make it 1D
    return F.binary_cross_entropy_with_logits(outputs.squeeze(1), targets.float(), reduction="elementwise_mean")
