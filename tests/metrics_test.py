import pyjet
from pyjet.metrics import accuracy
from pyjet.test_utils import ReluNet
import pyjet.backend as J
import numpy as np
import pytest

def test_accuracy():
    x = J.Variable(J.Tensor([[0.9, 0, 0.1, 0], [0.2, 0.3, 0.1, 0.4]]))
    y = J.Variable(J.Tensor([[1, 0, 0, 0], [1, 0, 0, 0]]))
    assert(accuracy(x.data.numpy(), y.data.numpy()) == 50.)

# def test_categorical_crossentropy():
#     x = J.Variable(J.Tensor([[0.5, 0, 0.5, 0]]))
#     y = J.Variable(J.Tensor([[1, 0, 0, 0]]))
#     np.testing.assert_almost_equal(0.6931471824645996, categorical_crossentropy(x, y).data[0])
