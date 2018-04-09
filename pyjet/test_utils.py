from .models import SLModel, loss_function
from .losses import categorical_crossentropy
from .metrics import accuracy
import torch.nn as nn
import torch.nn.functional as F

class ReluNet(SLModel):
    def __init__(self):
        super(ReluNet, self).__init__()
        self.register_loss_function(self.my_loss)
        self.register_metric_function(self.my_metric)

    def call(self, x):

        self.loss_in = F.relu(x)
        return self.loss_in

    def my_loss(self, targets):
        return categorical_crossentropy(self.loss_in, targets)

    def my_metric(self, targets):
        return accuracy(self.loss_in.data, targets.data)