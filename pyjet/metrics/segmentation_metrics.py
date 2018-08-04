import torch

from .abstract_metrics import AverageMetric
from ..registry import register_metric


class IOU(AverageMetric):
    """
    Computes the intersection over union over predictions. Only supports
    a single class.

    Args:
        None

    Returns:
        An IOU metric that can maintain its own internal state.

    Inputs:
        y_pred (torch.FloatTensor): A Float Tensor with the predicted
            probabilites for the output.
        y_true (torch.LongTensor): A Tensor with ground truth segmentation
    Outputs:
        A scalar tensor equal to the IOU of the y_pred
    """

    def score(self, y_pred, y_true):
        # Expect two tensors of the same shape
        assert tuple(y_pred.size()) == tuple(y_true.size())
        y_pred = y_pred > 0.5
        y_true = y_true > 0.5  # Casts the ground truth to a byte tensor

        non_batch_dims = tuple(range(1, y_pred.dim()))
        union = (y_pred | y_true).sum(non_batch_dims)
        intersection = (y_pred & y_true).sum(non_batch_dims)

        # If the union is 0, we both predicted nothing and ground
        # truth was nothing. All other no ground truth cases will
        # be 0.
        no_union = union == 0
        intersection = intersection.masked_fill_(no_union, 1.)
        union = union.masked_fill_(no_union, 1.)
        return torch.mean(intersection.float() / union.float())


iou = IOU()
register_metric('iou', iou)
