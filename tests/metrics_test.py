import torch.nn.functional as F

import pyjet
from pyjet.metrics import \
    (Accuracy,
     AccuracyWithLogits,
     TopKAccuracy)
import pyjet.backend as J
import numpy as np
import pytest


def test_accuracy():
    accuracy = Accuracy()
    accuracy_with_logits = AccuracyWithLogits()
    top2 = TopKAccuracy(2)
    top3 = TopKAccuracy(3)

    # First try with a multi class input
    x = J.Variable(J.Tensor([[0.9, 0, 0.1, 0], [0.2, 0.3, 0.1, 0.4]]))
    # Labels are the indicies
    y = J.Variable(J.LongTensor([0, 0]))
    assert accuracy(y, x).item() == 50.
    # Since applying the softmax again won't change the ordering
    assert accuracy_with_logits(y, x) == 50.
    assert top2(y, x) == 50.
    assert top3(y, x) == 100.

    # Now try with binary class input
    x_logit = J.Variable(J.Tensor([[100.], [-100.]]))
    x = F.sigmoid(x_logit)
    y = J.Variable(J.LongTensor([0, 0]))
    assert accuracy(y, x) == 50.
    assert accuracy_with_logits(y, x_logit) == 50.


def test_accumulation():
    accuracy = Accuracy()

    x = J.Variable(J.Tensor([[0.9, 0, 0.1, 0],
                             [0.2, 0.3, 0.1, 0.4],
                             [1.0, 0.0, 0.0, 0.0]]))
    y = J.Variable(J.LongTensor([1, 2, 3]))
    accuracy(y, x)
    x = J.Variable(J.Tensor([[0.9, 0, 0.1, 0], [0.2, 0.3, 0.1, 0.4]]))
    y = J.Variable(J.LongTensor([0, 0]))
    accuracy(y, x)
    assert accuracy.accumulate() == 20.
    accuracy = accuracy.reset()
    accuracy(y, x)
    assert accuracy.accumulate() == 50.
