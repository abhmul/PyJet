import os
import json
import logging
from functools import partial

import torch
from torch.autograd import Variable  # DO NOT REMOVE

from . import utils

# Config Settings
config = DEFAULT_CONFIG = dict(epsilon=1e-11, channels_mode="channels_last")
CONFIG_PATH = os.path.join(
    utils.safe_open_dir(os.path.expanduser("~/.pyjet/")), "pyjet.json"
)
# Create the config if it doesn't exist
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "w") as config_file:
        json.dump(DEFAULT_CONFIG, config_file)
# Load the config
with open(CONFIG_PATH, "r") as config_file:
    config = json.load(config_file)


def validate_config(config_dict):
    assert config_dict["channels_mode"] in {"channels_first", "channels_last"}, (
        "Channels mode must be either "
        "channels_first or channels_last, "
        "not {}".format(config_dict["channels_mode"])
    )


# Define the global backend variables
validate_config(config)
epsilon = config["epsilon"]
channels_mode = config["channels_mode"]
logging.info(
    "PyJet using config: epsilon={epsilon}, channels_mode"
    "={channels_mode}".format(**config)
)

# Set up the use of cuda if available
use_cuda = torch.cuda.device_count() > 0
device = torch.cuda.current_device() if use_cuda else torch.current_device()


def flatten(x):
    """Flattens along axis 0 (# rows in == # rows out)"""
    return x.view(x.size(0), -1)


def softmax(x):
    # BUG some shape error
    # .clamp(epsilon, 1.)
    normalized_exp = (x - x.max(1)[0].expand(*x.size())).exp()
    return normalized_exp / normalized_exp.sum(1).expand(*x.size())


def sum(x):
    """Sums a tensor long all non-batch dimensions"""
    return x.sum(tuple(range(1, x.dim())))


def zero_center(x):
    return x - x.mean()


def standardize(x):
    std = (x.pow(2).mean() - x.mean().pow(2)).sqrt()
    return zero_center(x) / std.expand(*x.size()).clamp(min=epsilon)


def from_numpy(x):
    return torch.from_numpy(x).cuda() if use_cuda else torch.from_numpy(x)


def to_numpy(x):
    return x.cpu().numpy() if use_cuda else x.numpy()


# TODO: Figure out a way to do this with python decorators
tensor = partial(torch.tensor, device=device)
FloatTensor = partial(tensor, dtype=torch.float)
DoubleTensor = partial(tensor, dtype=torch.double)
HalfTensor = partial(tensor, dtype=torch.half)
ByteTensor = partial(tensor, dtype=torch.uint8)
CharTensor = partial(tensor, dtype=torch.int8)
ShortTensor = partial(tensor, dtype=torch.short)
IntTensor = partial(tensor, dtype=torch.int)
LongTensor = partial(tensor, dtype=torch.long)
BoolTensor = partial(tensor, dtype=torch.bool)

Tensor = FloatTensor

# Tensor fillers
zeros = partial(torch.zeros, device=device)
ones = partial(torch.ones, device=device)
rand = partial(torch.rand, device=device)
arange = partial(torch.arange, device=device)

print("PyJet is using " + ("CUDA" if use_cuda else "CPU") + ".")
