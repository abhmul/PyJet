import torch
import torch.nn as nn
import torch.nn.functional as F

def nll_loss(y_pred, y_true, size_average=True):
    """
    y_pred -- B x L
    y_true -- B x L
    """
    y_true = torch.nonzero(y_true)[:, 1] # B x 2 -> B
    return F.nll_loss(y_pred, y_true, size_average=size_average)
