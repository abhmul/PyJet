import pyjet
from pyjet.metrics import accuracy
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


def test_test_batch(relu_net):
    x = np.array([[.2, .3, .5, -1], [-1, .6, .4, -1]])
    y = np.array([2, 2])
    # relu_net.add_loss_function(categorical_crossentropy)
    loss_dict = relu_net.test_on_batch(x, y)
    expected_loss = -(math.log(x[0, y[0]]) + math.log(x[1, y[1]])) / 2
    assert math.isclose(loss_dict["loss"], expected_loss, rel_tol=1e-7)
    # Now try with adding a metric
    relu_net.add_metric_function(accuracy)
    score_dict = relu_net.test_on_batch(x, y)

    assert math.isclose(score_dict["loss"], expected_loss, rel_tol=1e-7)
    assert score_dict["accuracy"] == 50.
