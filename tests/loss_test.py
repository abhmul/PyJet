from pyjet.losses import categorical_crossentropy
import pyjet.backend as J
import numpy as np
import pytest


def test_categorical_crossentropy():
    x = J.Variable(J.Tensor([[0.5, 0, 0.5, 0]]))
    y = J.Variable(J.LongTensor([0]))
    np.testing.assert_almost_equal(0.6931471824645996,
                                   categorical_crossentropy(x, y).item())
