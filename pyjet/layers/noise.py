import logging

from torch.autograd import Variable

from . import layer
from .. import backend as J


class GaussianNoise1D(layer.Layer):

    def __init__(self, size, std=0.05):
        super().__init__()
        self.noise = Variable(J.zeros(1, 1, size), requires_grad=False)
        self.std = std
        self.__descriptor = "{name}(size={size}, std={std})".format(name=self.__class__.__name__, size=size, std=std)
        logging.info("Creating layer %r" % self)

    def forward(self, x):
        if not self.training:
            return x
        self.noise.data.normal_(0, std=self.std)
        return x + self.noise

    def __str__(self):
        return self.__descriptor
