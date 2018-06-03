import torch.nn.functional as F

import pyjet
from pyjet.metrics import \
    (Accuracy,
     AccuracyWithLogits,
     TopKAccuracy,
     AverageMetric)
import pyjet.backend as J
import numpy as np
import pytest


def test_accuracy():
    accuracy = Accuracy()
    accuracy_with_logits = AccuracyWithLogits()
    top2 = TopKAccuracy(2)
    top3 = TopKAccuracy(3)
    fake_accuracy = AverageMetric(accuracy.score)

    # First try with a multi class input
    x = J.Variable(J.Tensor([[0.9, 0, 0.1, 0], [0.2, 0.3, 0.1, 0.4]]))
    # Labels are the indicies
    y = J.Variable(J.LongTensor([0, 0]))
    assert accuracy(x, y).item() == 50.
    assert fake_accuracy(x, y).item() == 50.
    # Since applying the softmax again won't change the ordering
    assert accuracy_with_logits(x, y) == 50.
    assert top2(x, y) == 50.
    assert top3(x, y) == 100.

    # Now try with binary class input
    x_logit = J.Variable(J.Tensor([[100.], [-100.]]))
    x = F.sigmoid(x_logit)
    y = J.Variable(J.LongTensor([0, 0]))
    assert accuracy(x, y) == 50.
    assert fake_accuracy(x, y) == 50.
    assert accuracy_with_logits(x_logit, y) == 50.


def test_accumulation():
    accuracy = Accuracy()

    x = J.Variable(J.Tensor([[0.9, 0, 0.1, 0],
                             [0.2, 0.3, 0.1, 0.4],
                             [1.0, 0.0, 0.0, 0.0]]))
    y = J.Variable(J.LongTensor([1, 2, 3]))
    accuracy(x, y)
    x = J.Variable(J.Tensor([[0.9, 0, 0.1, 0], [0.2, 0.3, 0.1, 0.4]]))
    y = J.Variable(J.LongTensor([0, 0]))
    accuracy(x, y)
    assert accuracy.accumulate() == 20.
    accuracy = accuracy.reset()
    accuracy(x, y)
    assert accuracy.accumulate() == 50.
