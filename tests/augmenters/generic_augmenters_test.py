import numpy as np
from pyjet.test_utils import ZeroAugmenter
from pyjet.data import NpDataset
import pytest


def test_augmenter_basic():
    # Try different combinations of with labels and without
    data = NpDataset(x=np.ones((32, )), y=np.ones((32, )))
    augmenter = ZeroAugmenter(labels=False, augment_labels=False)
    assert not augmenter.labels
    assert not augmenter.augment_labels
    data.output_labels = False
    x = next(augmenter(data.flow(batch_size=32)))
    assert np.all(x == 0.)

    augmenter = ZeroAugmenter(labels=False, augment_labels=True)
    assert not augmenter.labels
    assert augmenter.augment_labels
    data.output_labels = False
    x = next(augmenter(data.flow(batch_size=32)))
    assert np.all(x == 0.)

    augmenter = ZeroAugmenter(labels=True, augment_labels=False)
    assert augmenter.labels
    assert not augmenter.augment_labels
    data.output_labels = True
    x, y = next(augmenter(data.flow(batch_size=32)))
    assert np.all(x == 0.)
    assert np.all(y == 1.)

    augmenter = ZeroAugmenter(labels=True, augment_labels=True)
    assert augmenter.labels
    assert augmenter.augment_labels
    data.output_labels = True
    x, y = next(augmenter(data.flow(batch_size=32)))
    assert np.all(x == 0.)
    assert np.all(y == 0.)

    # Try a generic python generator
    def datagen():
        yield np.ones((32, ))

    augmenter = ZeroAugmenter(labels=False, augment_labels=False)
    x = next(augmenter(datagen()))
    assert np.all(x == 0.)
