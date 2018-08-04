from .models import SLModel
from . import layers
from .augmenters import Augmenter
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReluNet(SLModel):

    def __init__(self):
        super(ReluNet, self).__init__()
        self.param = nn.Linear(100, 100)

    def forward(self, x):
        self.loss_in = F.relu(x)
        return self.loss_in

class InferNet1D(SLModel):

    def __init__(self):
        super(InferNet1D, self).__init__()
        self.c1 = layers.Conv1D(10, kernel_size=3)
        self.r11 = layers.GRU(5, return_sequences=True)
        self.r12 = layers.SimpleRNN(5, return_sequences=True)
        self.r13 = layers.LSTM(5, return_sequences=True)
        self.att1 = layers.ContextAttention(5)
        self.flatten = layers.Flatten()
        self.fc = layers.FullyConnected(2)
        self.infer_inputs(layers.Input(10, 3))

    def forward(self, x):
        x = self.c1(x)
        x = torch.cat([self.r11(x), self.r12(x), self.r13(x)], dim=-1)
        x = self.att1(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class InferNet2D(SLModel):
    def __init__(self):
        super(InferNet2D, self).__init__()
        self.c1 = layers.Conv2D(10, kernel_size=3)
        self.flatten = layers.Flatten()
        self.fc = layers.FullyConnected(2)
        self.infer_inputs(layers.Input(10, 10, 3))

    def forward(self, x):
        x = self.c1(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class InferNet3D(SLModel):
    def __init__(self):
        super(InferNet3D, self).__init__()
        self.c1 = layers.Conv3D(10, kernel_size=3)
        self.flatten = layers.Flatten()
        self.fc = layers.FullyConnected(2)
        self.infer_inputs(layers.Input(10, 10, 10, 3))

    def forward(self, x):
        x = self.c1(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ZeroAugmenter(Augmenter):

    def augment(self, x):
        return 0. * x


def binary_loss(y_pred, y_true):
    return (y_pred * y_true).sum()


def one_loss(y_pred, y_true):
    return binary_loss(y_pred, y_true) * 0. + 1.


def multi_binary_loss(y_pred1, y_pred2, y_true):
    return binary_loss(y_pred1, y_true) + binary_loss(y_pred2, y_true)
