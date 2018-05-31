import torch
import torch.nn.functional as F


class Metric(object):
    def __init__(self, metric_func=None):
        self.metric_func = metric_func

    def __call__(self, y_true, y_pred):
        if self.metric_func is not None:
            return self.metric_func(y_true, y_pred)
        else:
            raise NotImplementedError()

    def accumulate(self, value=None):
        pass

    def reset(self):
        return self.__class__(metric_func=self.metric_func)


class AverageMetric(Metric):
    def __init__(self, metric_func=None):
        super(AverageMetric, self).__init__(metric_func=metric_func)
        self.metric_sum = 0.
        self.count = 0

    def accumulate(self, value):
        self.metric_sum += value
        self.metric_count += 1
        return self.metric_sum / self.metric_count


class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.correct = 0
        self.total = 0

    def __call__(self, y_true, y_pred):
        # Expect output and target to be B x 1 or B x C or target can be
        # B (with ints from 0 to C-1)
        assert y_pred.dim() == 2, "y_pred should be a 2-dimensional tensor"
        assert y_pred.size(0) == y_true.size(
            0), "y_preds and y_trues should have the same number of samples"
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

        # Accumulate the statistics
        self.correct += correct
        self.total += total
        return (correct / total) * 100.

    def accumulate(self, value=None):
        return self.correct / self.total


class AccuracyWithLogits(Accuracy):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        if y_pred.dim() == 2 and y_pred.size(1) > 1:
            y_pred = F.softmax(y_pred, dim=1)
        else:
            y_pred = F.sigmoid(y_pred)
        return super().__call__(y_true, y_pred)


class TopKAccuracy(Accuracy):
    """Computes the precision@k for the specified values of k

    # Arguments
        k -- The k to calculate precision@k (topk accuracy)
    # Inputs
        y_true -- A 1D torch LongTensor of the correct classes
        y_pred -- A 2D Float Tensor with the predicted probabilites for each
                  class.
    # Outputs:
        A scalar tensor equal to the topk accuracy of the y_pred

    """

    def __init__(self, k=3):
        super().__init__()
        self.k = k

    def __call__(self, y_true, y_pred):
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
        self.correct += correct_k
        self.total += batch_size
        return 100.0 * correct_k / batch_size
