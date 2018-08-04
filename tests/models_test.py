import torch.optim as optim

import pyjet
from pyjet.metrics import Accuracy
from pyjet.losses import categorical_crossentropy
from pyjet.test_utils import ReluNet, binary_loss, multi_binary_loss, \
    one_loss, InferNet1D, InferNet2D, InferNet3D
import pyjet.backend as J
import math
import numpy as np
import pytest


@pytest.fixture
def relu_net():
    net = ReluNet()
    return net

@pytest.fixture
def test_infer_net1d():
    net = InferNet1D()
    return net

@pytest.fixture
def test_infer_net2d():
    net = InferNet2D()
    return net

@pytest.fixture
def test_infer_net3d():
    net = InferNet3D()
    return net

@pytest.fixture
def binary_loss_fn():
    return binary_loss


@pytest.fixture
def multi_binary_loss_fn():
    return multi_binary_loss

def test_infer_net(test_infer_net1d, test_infer_net2d, test_infer_net3d):
    test_infer_net1d(J.zeros(1, 10, 3))
    test_infer_net2d(J.zeros(1, 10, 10, 3))
    test_infer_net3d(J.zeros(1, 10, 10, 10, 3))

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


def test_optimizer(relu_net):
    optimizer = optim.SGD(relu_net.parameters(), lr=0.01)

    # Test by adding an unnamed optimizer
    relu_net.add_optimizer(optimizer)
    assert len(relu_net.optimizer_manager.optimizers) == \
        len(relu_net.optimizer_manager.names) == 1
    assert optimizer in relu_net.optimizer_manager.optimizers
    assert "optimizer_0" in relu_net.optimizer_manager.names

    # Test by adding a named optimizer
    name = "sgd_optim"
    relu_net.clear_optimizers()
    relu_net.add_optimizer(optimizer, name=name)
    assert len(relu_net.optimizer_manager.optimizers) == \
        len(relu_net.optimizer_manager.names) == 1
    assert optimizer in relu_net.optimizer_manager.optimizers
    assert name in relu_net.optimizer_manager.names

    # Test by adding multiple optimizers
    optimizer2 = optim.SGD(relu_net.parameters(), lr=0.02)
    relu_net.clear_optimizers()
    relu_net.add_optimizer(optimizer, name=name)
    relu_net.add_optimizer(optimizer2)
    assert len(relu_net.optimizer_manager.optimizers) == \
        len(relu_net.optimizer_manager.names) == 2
    assert optimizer in relu_net.optimizer_manager.optimizers
    assert optimizer2 in relu_net.optimizer_manager.optimizers
    assert name in relu_net.optimizer_manager.names
    assert "optimizer_1" in relu_net.optimizer_manager.names

    # Test removing an optimizer
    optim_info = relu_net.remove_optimizer()
    assert optim_info["name"] == "optimizer_1"
    assert optim_info["optimizer"] is optimizer2
    assert len(relu_net.optimizer_manager.optimizers) == \
        len(relu_net.optimizer_manager.names) == 1
    # Add an optimizer to check removing
    # mamed optimizers works out of order
    relu_net.add_optimizer(optimizer)
    optim_info = relu_net.remove_optimizer(name=name)
    assert optim_info["name"] == name
    assert optim_info["optimizer"] is optimizer
    assert len(relu_net.optimizer_manager.optimizers) == \
        len(relu_net.optimizer_manager.names) == 1
    relu_net.clear_optimizers()


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
    relu_net.add_loss(one_loss, inputs=['loss_in2'], name="new_loss")
    pred = relu_net(x_torch)
    relu_net.loss_in2 = pred
    assert relu_net.loss(y_torch) == 5.
    # Check that the individual loss values were also calculated
    assert relu_net.loss_manager.get_loss_score('loss_0') == 4.
    assert relu_net.loss_manager.get_loss_score('new_loss') == 1.

    # Now try removing the loss
    loss_info = relu_net.remove_loss(name="new_loss")
    assert loss_info["name"] == "new_loss"
    assert loss_info["loss"] == one_loss
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

    relu_net.clear_losses()
