import pyjet
from pyjet.losses import nll_loss
from pyjet.metrics import accuracy
from pyjet.models import SLModel
from pyjet.test_utils import ReluNet
import numpy as np
import pytest

@pytest.fixture
def relu_net():
    net = ReluNet()
    return net

def test_predict_batch(relu_net):
    x = np.array([[-1, 1, 2, -2]])
    expected = np.array([[0., 1., 2., 0.]])
    np.testing.assert_array_equal(relu_net.predict_on_batch(x), expected)

def test_evaluate_batch(relu_net):
    x = np.array([[-1, 1, 2, -2], [-2, 3, 1, 2]])
    y = np.array([[0, 0, 1, 0], [0, 0, 1, 0]])
    loss, metrics = relu_net.evaluate_on_batch(x, y, loss_fn=nll_loss, metrics=[accuracy])
    assert(loss == -1.5)
    assert(len(metrics) == 1)
    assert(metrics[0] == 50.)
