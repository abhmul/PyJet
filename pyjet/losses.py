import torch
import torch.nn as nn
import torch.nn.functional as F

import pyjet.backend as J

def categorical_crossentropy(outputs, targets, size_average=True):
    """
    y_pred -- B x L
    y_true -- B
    """
    return F.nll_loss(outputs.clamp(J.epsilon, 1 - J.epsilon).log(), targets, size_average=size_average)
