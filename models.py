import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import backend as J
from data import NpDataset, Dataset, DatasetGenerator, BatchPyGenerator, BatchGenerator
import numpy as np

def _serialize_input(*args, steps_per_epoch=None, batch_size=None, **kwargs):
    if len(args) == 1 and isinstance(args[0], BatchGenerator):
        return args[0]
    if len(args) == 1 and isinstance(args[0], Dataset):
        return DatasetGenerator(args[0],  steps_per_epoch, batch_size, **kwargs)
    if len(args) == 2 and np.all([isinstance(x, np.ndarray) for x in args]):
        return DatasetGenerator(NpDataset(*args), steps_per_epoch, batch_size, **kwargs)
    if len(args) == 1 and (hasattr(args[0], 'next') or hasattr(args[0], '__next__')):
        return BatchPyGenerator(args, steps_per_epoch)
    # TODO Serialize more inputs
    if np.all([isinstance(x, Dataset) for x in args]):
        raise NotImplementedError
    else:
        raise ValueError("Could not serialize input to model")

# TODO Not sure whether I'll need to seperate RL models and SL models. Hopefully
# I planned this out right
class SLModel(nn.Module):

    def train_on_batch(self, ):

    # TODO Problems with this?
    def predict_on_batch(self, inputs, **kwargs):
        """
        Will add more to this documentation

        Inputs must be castable to Torch Tensors (list, numpy array, etc.)
        The result from the model must be one variable. [PROBLEM]
        kwargs can encode other non tensor info
        """
        result = self(*[Variable(J.Tensor(inp)) for inp in inputs], **kwargs):
        return result.data.numpy()
