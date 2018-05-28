from .models import SLModel
from .augmenters import Augmenter
import torch.nn.functional as F


class ReluNet(SLModel):
    def forward(self, x):
        return F.relu(x)


class ZeroAugmenter(Augmenter):

    def augment(self, x):
        return 0. * x
