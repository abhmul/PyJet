from .models import SLModel
from .augmenters import Augmenter
import torch.nn as nn
import torch.nn.functional as F


class ReluNet(SLModel):

    def __init__(self):
        super(ReluNet, self).__init__()
        self.param = nn.Linear(100, 100)

    def forward(self, x):
        self.loss_in = F.relu(x)
        return self.loss_in


class ZeroAugmenter(Augmenter):

    def augment(self, x):
        return 0. * x


def binary_loss(y_pred, y_true):
    return (y_pred * y_true).sum()


def multi_binary_loss(y_pred1, y_pred2, y_true):
    return binary_loss(y_pred1, y_true) + binary_loss(y_pred2, y_true)
