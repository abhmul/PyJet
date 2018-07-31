import os
import json
import logging

import torch

from . import utils

# Config Settings
config = DEFAULT_CONFIG = dict(epsilon=1e-11, channels_mode="channels_last")
CONFIG_PATH = os.path.join(utils.safe_open_dir(
    os.path.expanduser("~/.pyjet/")), "pyjet.json")
# Create the config if it doesn't exist
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'w') as config_file:
        json.dump(DEFAULT_CONFIG, config_file)
# Load the config
with open(CONFIG_PATH, 'r') as config_file:
    config = json.load(config_file)


def validate_config(config_dict):
    assert config_dict["channels_mode"] in \
        {"channels_first", "channels_last"}, "Channels mode must be either " \
        "channels_first or channels_last, " \
        "not {}".format(config_dict["channels_mode"])


# Define the global backend variables
validate_config(config)
epsilon = config['epsilon']
channels_mode = config["channels_mode"]
logging.info("PyJet using config: epsilon={epsilon}, channels_mode"
             "={channels_mode}".format(**config))

# Set up the use of cuda if available
use_cuda = torch.cuda.is_available()


def cudaFloatTensor(x):
    # Optimization for casting things to cuda tensors
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


def to_numpy(x):
    return x.cpu().numpy() if use_cuda else x.numpy()


def arange(start, end=None, step=1, out=None):
    if end is None:
        x = torch.arange(0, start, step=step, out=out)
    else:
        x = torch.arange(start, end, step=step, out=out)
    return x.cuda() if use_cuda else x


def rand(*sizes, out=None):
    x = torch.rand(*sizes, out=out)
    return x.cuda() if use_cuda else x


# use_cuda = False
FloatTensor = cudaFloatTensor if use_cuda else torch.FloatTensor
LongTensor = cudaLongTensor if use_cuda else torch.LongTensor
ByteTensor = cudaByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
# Tensor fillers
zeros = cudaZeros if use_cuda else torch.zeros
ones = cudaOnes if use_cuda else torch.ones

print("PyJet is using " + ("CUDA" if use_cuda else "CPU") + ".")
