import torch
import torch.nn.functional as F

from .abstract_metrics import AverageMetric
from ..registry import register_metric


class Accuracy(AverageMetric):
    """
    Computes the accuracy over predictions

    Args:
        None

    Returns:
        An accuracy metric that can maintain its own internal state.

    Inputs:
        y_pred (torch.FloatTensor): A 2D Float Tensor with the predicted
            probabilites for each class.
        y_true (torch.LongTensor): A 1D torch LongTensor of the correct classes
    Outputs:
        A scalar tensor equal to the accuracy of the y_pred

    """
    def score(self, y_pred, y_true):
        # Expect output and target to be B x 1 or B x C or target can be
        # B (with ints from 0 to C-1)
        assert y_pred.dim() == 2, "y_pred should be a 2-dimensional tensor"
        total = y_true.size(0)
        # Turn it into a 1d class encoding
        if y_true.dim() == 2:
            if y_true.size(1) > 1:
                raise NotImplementedError(
                    "Multiclass with 2d targets is not impelemented yet")
            y_true = y_true.squeeze(1).long()

        # Change the y_pred to have two cols if it only has 1
        if y_pred.size(1) == 1:
            # Want to consider the 0.5 case
            y_pred = (y_pred >= 0.5).float()
            y_pred = torch.cat([1 - y_pred, y_pred], dim=1)

        # Compute the accuracy
        _, predicted = torch.max(y_pred, 1)
        correct = (predicted == y_true).float().sum(0)

        return (correct / total) * 100.


class AccuracyWithLogits(Accuracy):

    """An accuracy metric that takes as input the logits. See `Accuracy` for
    more details.
    """
    def score(self, y_pred, y_true):
        if y_pred.dim() == 2 and y_pred.size(1) > 1:
            y_pred = F.softmax(y_pred, dim=1)
        else:
            y_pred = torch.sigmoid(y_pred)
        return super().score(y_pred, y_true)


class TopKAccuracy(Accuracy):
    """Computes the precision@k for the specified values of k

    Args:
        k (int): The k to calculate precision@k (topk accuracy)

    Returns:
        A TopKAccuracy metric that can maintain its own internal state.

    Inputs:
        y_pred (torch.FloatTensor) A 2D Float Tensor with the predicted
            probabilites for each class.
        y_true (torch.LongTensor) A 1D torch LongTensor of the correct classes

    Outputs:
        A scalar tensor equal to the topk accuracy of the y_pred

    """
    def __init__(self, k=3):
        super().__init__()
        self.k = k

    def score(self, y_pred, y_true):
        assert y_true.dim() == y_pred.dim() - 1 == 1
        channel_dim = 1
        batch_size = y_true.size(0)
        # Check to see if there's only 1 channel (then its binary
        # classification)
        if y_pred.size(channel_dim) == 1:
            y_pred = y_pred.squeeze(channel_dim)  # B x ...
            # 2 x B x ... -> B x 2 x ...
            y_pred = y_pred.stack([1 - y_pred, y_pred]).t()

        # Get the indicies of the topk along the channel dim
        _, pred = y_pred.topk(self.k, channel_dim, True, True)  # B x k x ...
        pred = pred.t()  # k x B x ...
        # target: B -> 1 x B -> k x B x ...
        correct = pred.eq(y_true.view(1, -1).expand_as(pred))

        correct_k = correct[:self.k].view(-1).float().sum(0)

        # Accumulate results
        return 100. * correct_k / batch_size


accuracy = Accuracy()
accuracy_with_logits = AccuracyWithLogits()
top2_accuracy = TopKAccuracy(2)
top3_accuracy = TopKAccuracy(3)
top5_accuracy = TopKAccuracy(5)

register_metric('accuracy', accuracy)
register_metric('accuracy_with_logits', accuracy_with_logits)
register_metric('top2_accuracy', top2_accuracy)
register_metric('top3_accuracy', top3_accuracy)
register_metric('top5_accuracy', top5_accuracy)
