import torch
from torch.autograd import Variable
import torch.nn.functional as F

epsilon = 1e-11

# Optimization for casting things to cuda tensors

# Set up the use of cuda if available
use_cuda = torch.cuda.is_available()

def cudaFloatTensor(x):
    return torch.FloatTensor(x).cuda()


def cudaLongTensor(x):
    return torch.LongTensor(x).cuda()


def cudaByteTensor(x):
    return torch.ByteTensor(x).cuda()


def cudaZeros(*args):
    return torch.zeros(*args).cuda()

def cudaOnes(*args):
    return torch.ones(*args).cuda()


def flatten(x):
    """Flattens along axis 0 (# rows in == # rows out)"""
    return x.view(x.size(0), -1)


def softmax(x):
    # BUG some shape error
    # .clamp(epsilon, 1.)
    normalized_exp = (x - x.max(1)[0].expand(*x.size())).exp()
    return normalized_exp / normalized_exp.sum(1).expand(*x.size())


def zero_center(x):
    return x - x.mean()


def standardize(x):
    std = (x.pow(2).mean() - x.mean().pow(2)).sqrt()
    return zero_center(x) / std.expand(*x.size()).clamp(min=epsilon)


def from_numpy(x):
    return torch.from_numpy(x).cuda() if use_cuda else torch.from_numpy(x)


# use_cuda = False
FloatTensor = cudaFloatTensor if use_cuda else torch.FloatTensor
LongTensor = cudaLongTensor if use_cuda else torch.LongTensor
ByteTensor = cudaByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
# Tensor fillers
zeros = cudaZeros if use_cuda else torch.zeros
ones = cudaOnes if use_cuda else torch.ones

print("PyJet is using " + ("CUDA" if use_cuda else "CPU") + ".")
