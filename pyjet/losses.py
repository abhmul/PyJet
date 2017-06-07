import torch
import torch.nn as nn
import torch.nn.functional as F

def nll_loss(y_pred, y_true, size_average=True):
    """
    y_pred -- B x L
    y_true -- B x L, must be 1 or 0
    """
    return torch.sum(-y_pred[y_true.byte()]) / (y_true.size(0) if size_average else 1.)

def categorical_crossentropy(y_pred, y_true, size_average=True):
    """
    y_pred -- B x L
    y_true -- B x L
    """
    return torch.sum(-torch.log(y_pred[y_true.byte()])) / (y_true.size(0) if size_average else 1.)
