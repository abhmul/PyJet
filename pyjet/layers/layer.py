import torch.nn as nn


class Layer(nn.Module):

    def __init__(self):
        super(Layer, self).__init__()
        self.built = False

    def forward(self, *input):
        raise NotImplementedError()

    def reset_parameters(self):
        pass

    def __str__(self):
        return self.__class__.__name__ + "()"

    def __repr__(self):
        return str(self)
