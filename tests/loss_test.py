import pyjet
from pyjet.models import SLModel
from pyjet.losses import nll_loss, categorical_crossentropy
from pyjet.test_utils import ReluNet
import pyjet.backend as J
import numpy as np
import pytest

def test_nll_loss():
    x = J.Variable(J.Tensor([[0.5, 0, 0.5, 0]]))
    y = J.Variable(J.Tensor([[1, 0, 0, 0]]))
    assert(nll_loss(x, y) == -0.5)

def test_categorical_crossentropy():
    x = J.Variable(J.Tensor([[0.5, 0, 0.5, 0]]))
    y = J.Variable(J.Tensor([[1, 0, 0, 0]]))
    np.testing.assert_almost_equal(0.6931471824645996, categorical_crossentropy(x, y).data[0])
