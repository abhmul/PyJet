import logging

import torch
import torch.nn as nn

from . import functions as L


def build_strided_pool(name, kernel_size, stride=None, padding=1, dilation=1):

    layer = StridedPool.pool_funcs[name](kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    logging.info("Creating layer: %r" % layer)
    return layer


class StridedPool(nn.Module):

    pool_funcs = {"max1d": nn.MaxPool1d,
                  "max2d": nn.MaxPool2d,
                  "max3d": nn.MaxPool3d,
                  "avg1d": nn.AvgPool1d,
                  "avg2d": nn.AvgPool2d,
                  "avg3d": nn.AvgPool3d}

    def __init__(self, pool_type, kernel_size, stride=None, padding='same', dilation=1):
        super(StridedPool, self).__init__()
        padding = (kernel_size - 1) // 2 if padding == 'same' else padding
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.pool = build_strided_pool(pool_type, kernel_size, stride=stride, padding=padding, dilation=dilation)

    def calc_output_size(self, input_size):
        output_size = (input_size - self.dilation * (self.kernel_size - 1) + 2 * self.padding - 1) // self.stride + 1
        return output_size

    def forward(self, x):
        # Expect x as BatchSize x Filters x Length1 x ... x LengthN
        return self.pool(x)

    def __str__(self):
        return "%r" % self.pool

    def __repr__(self):
        return str(self)


class MaxPooling1D(StridedPool):

    def __init__(self, kernel_size, stride=None, padding='same', dilation=1):
        super(MaxPooling1D, self).__init__("max1d", kernel_size, stride=stride, padding=padding, dilation=dilation)


class AveragePooling1D(StridedPool):

    def __init__(self, kernel_size, stride=None, padding='same', dilation=1):
        super(AveragePooling1D, self).__init__("avg1d", kernel_size, stride=stride, padding=padding, dilation=dilation)


class GlobalMaxPooling1D(nn.Module):

    def __init__(self):
        super(GlobalMaxPooling1D, self).__init__()

        # Logging
        logging.info("Creating layer: {}".format(str(self)))

    def forward(self, x):
        # The input comes in as B x L x E
        return torch.max(x, dim=1)[0]

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class GlobalAveragePooling1D(nn.Module):

    def __init__(self):
        super(GlobalAveragePooling1D, self).__init__()

        # Logging
        logging.info("Creating layer: {}".format(str(self)))

    def forward(self, x):
        # The input comes in as B x L x E
        return torch.mean(x, dim=1)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class KMaxPooling1D(nn.Module):

    def __init__(self, k):
        super(KMaxPooling1D, self).__init__()
        self.k = k

        # Logging
        logging.info("Creating layer: {}".format(str(self)))

    def forward(self, x):
        # B x L x E
        return L.kmax_pooling(x, 1, self.k)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return self.__class__.__name__ + "(k=%s)" % self.k
