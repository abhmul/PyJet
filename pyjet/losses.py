import torch
import torch.nn as nn
import torch.nn.functional as F

def categorical_crossentropy(outputs, targets, size_average=True):
    """
    y_pred -- B x L
    y_true -- B
    """
    return F.nll_loss(outputs.log(), targets, size_average=size_average)
