import pyjet
from pyjet.metrics import Accuracy
from pyjet.losses import categorical_crossentropy
from pyjet.test_utils import ReluNet, binary_loss, multi_binary_loss
import math
import numpy as np
import pytest


@pytest.fixture
def relu_net():
    net = ReluNet()
    return net


@pytest.fixture
def binary_loss_fn():
    return binary_loss


@pytest.fixture
def multi_binary_loss_fn():
    return multi_binary_loss


def test_predict_batch(relu_net):
    x = np.array([[-1., 1., 2., -2.]])
    expected = np.array([[0., 1., 2., 0.]])
    assert np.all(relu_net.predict_on_batch(x) == expected)


def test_validate_batch(relu_net):
    x = np.array([[.2, .3, .5, -1], [-1, .6, .4, -1]])
    y = np.array([2, 2])
    (loss, accuracy_score), preds = relu_net.validate_on_batch(
        x, y, metrics=[categorical_crossentropy, Accuracy()])
    expected_loss = -(math.log(x[0, y[0]]) + math.log(x[1, y[1]])) / 2
    assert math.isclose(loss, expected_loss, rel_tol=1e-7)
    assert accuracy_score == 50.


def test_loss(relu_net, binary_loss_fn, multi_binary_loss_fn):
    x = np.array([[-1., 1., 0., 1.],
                  [-2., 0., 1., 1.]])
    y = np.array([[1., 1., 0., 1.],
                  [1., 0., 1., 1.]])
    x_torch = relu_net.cast_input_to_torch(x)
    y_torch = relu_net.cast_target_to_torch(y)

    # Test it without passing it inputs
    assert len(relu_net.loss_manager) == 0
    relu_net.add_loss(binary_loss_fn)
    pred = relu_net(x_torch)
    assert relu_net.loss(y_torch) == 4.

    # Test it it with an input of 'loss_in'
    relu_net.clear_losses()
    assert len(relu_net.loss_manager) == 0
    relu_net.add_loss(binary_loss_fn, inputs='loss_in')
    pred = relu_net(x_torch)
    assert relu_net.loss(y_torch) == 4.

    # Test it it with an input that's not 'loss_in' and using multi loss
    relu_net.add_loss(binary_loss_fn, inputs=['loss_in2'], name="new_loss")
    pred = relu_net(x_torch)
    relu_net.loss_in2 = pred
    assert relu_net.loss(y_torch) == 8.
    # Check that the individual loss values were also calculated
    assert relu_net.loss_manager.get_loss_score('loss_0') == 4.
    assert relu_net.loss_manager.get_loss_score('new_loss') == 4.

    # Now try removing the loss
    loss_info = relu_net.remove_loss(name="new_loss")
    assert loss_info["name"] == "new_loss"
    assert loss_info["loss"] == binary_loss_fn
    assert loss_info["inputs"] == ["loss_in2"]
    assert loss_info["weight"] == 1.0

    # Pop the last loss
    loss_info = relu_net.remove_loss()
    assert loss_info["name"] == "loss_0"
    assert loss_info["loss"] == binary_loss_fn
    assert loss_info["inputs"] == ["loss_in"]
    assert loss_info["weight"] == 1.0

    # Now try adding a multi input loss
    relu_net.add_loss(multi_binary_loss_fn, inputs=['loss_in', 'loss_in2'])
    pred = relu_net(x_torch)
    relu_net.loss_in2 = pred
    assert relu_net.loss(y_torch) == 8.
