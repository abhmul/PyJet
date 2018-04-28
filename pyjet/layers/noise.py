import logging

from torch.autograd import Variable

from . import layer
from .. import backend as J


class GaussianNoise1D(layer.Layer):

    def __init__(self, std=0.05, augment_prob=1.0):
        super().__init__()
        self.std = std
        self.augment_prob = augment_prob
        self.noise_size = tuple()
        self.noise = None
        self.mask_sample = None
        self.__descriptor = "{name}(std={std}, augment_prob={augment_prob})".format(name=self.__class__.__name__, std=std, augment_prob=augment_prob)
        logging.info("Creating layer %r" % self)

    def forward(self, x):
        if not self.training:
            return x
        self.init_noise(x)

        if self.augment_prob != 1.0:
            # 0 out the elements we don't want to change
            self.noise.data.masked_fill_(self.mask_sample > self.augment_prob, 0.)

        return x + self.noise

    def init_noise(self, x):
        # Create the noise (w/ mem optimization)
        x_shape = tuple(x.size())
        if self.noise_size != x_shape:
            self.noise = Variable(J.zeros(*x_shape), requires_grad=False)
            self.mask_sample = None if self.augment_prob == 1.0 else J.rand(*x_shape[:-1]).unsqueeze(-1)
            self.noise_size = x_shape
        else:
            self.mask_sample.uniform_()
        self.noise.data.normal_(0, std=self.std)

    def __str__(self):
        return self.__descriptor
