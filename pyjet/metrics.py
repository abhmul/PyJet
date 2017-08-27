import numpy as np


def topk_accuracy(output, target, topk):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    # output = output.data
    # target = target.data

    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:topk].view(-1).float().sum(0)
    res = correct_k.mul_(100.0 / batch_size)
    return res


def accuracy(output, target):
    return topk_accuracy(output, target, topk=1)
