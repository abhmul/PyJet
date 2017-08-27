import numpy as np
import pyjet.backend as J
import pytest


def test_flatten_3d_tensor():
    test_array = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)
    expected_array = np.arange(2 * 3 * 4).reshape(2, 3 * 4).astype(np.float32)
    test_tensor = J.Tensor(test_array)
    actual_array = J.flatten(test_tensor).numpy()
    np.testing.assert_array_equal(actual_array, expected_array)
