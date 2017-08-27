import pyjet
from pyjet.metrics import accuracy
from pyjet.test_utils import ReluNet
import pyjet.backend as J
import numpy as np
import pytest


def test_accuracy():
    x = J.Variable(J.Tensor([[0.9, 0, 0.1, 0], [0.2, 0.3, 0.1, 0.4]]))
    # Labels are the indicies
    y = J.Variable(J.LongTensor([0, 0]))
    assert(accuracy(x, y).data[0] == 50.)
