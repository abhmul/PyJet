from .models import SLModel
import torch.nn as nn
import torch.nn.functional as F

class ReluNet(SLModel):
    def forward(self, x):
        return F.relu(x)
