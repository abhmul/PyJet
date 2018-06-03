import pyjet
# from pyjet.metrics import accuracy
from pyjet.losses import categorical_crossentropy
from pyjet.test_utils import ReluNet
import math
import numpy as np
import pytest


@pytest.fixture
def relu_net():
    net = ReluNet()
    return net


def test_predict_batch(relu_net):
    x = np.array([[-1., 1., 2., -2.]])
    expected = np.array([[0., 1., 2., 0.]])
    assert np.all(relu_net.predict_on_batch(x) == expected)


# def test_validate_batch(relu_net):
#     x = np.array([[.2, .3, .5, -1], [-1, .6, .4, -1]])
#     y = np.array([2, 2])
#     (loss, accuracy_score), preds = relu_net.validate_on_batch(
#         x, y, metrics=[categorical_crossentropy, accuracy])
#     expected_loss = -(math.log(x[0, y[0]]) + math.log(x[1, y[1]])) / 2
#     assert math.isclose(loss, expected_loss, rel_tol=1e-7)
#     assert accuracy_score == 50.
