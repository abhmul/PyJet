import torch
import torch.nn.functional as F

def topk_accuracy(output, target, topk):
    """Computes the precision@k for the specified values of k

    # Arguments
        outputs -- A torch FloatTensor of arbitrary shape
        targets -- A torch LongTensor of the same size except along
                   the channels dimension (the target dimension - 1)
        topk -- The k to compute the topk accuracy for
    # Returns:
        A scalar tensor equal to the topk accuracy of the output

    """
    channel_dim = 1
    batch_size = target.size(0)
    # Check to see if there's only 1 channel (then its binary classification)
    if output.size(channel_dim) == 1:
        output = output.squeeze(channel_dim)  # B x ...
        # 2 x B x ... -> B x 2 x ...
        output = torch.stack([1 - output, output]).t()

    # Get the indicies of the topk along the channel dim
    _, pred = output.topk(topk, channel_dim, True, True)  # B x k x ...
    pred = pred.t()  # k x B x ...
    # target: B -> 1 x B -> k x B x ...
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:topk].view(-1).float().sum(0)
    res = correct_k.mul_(100.0 / batch_size)
    return res


def accuracy(output, target):
    # Expect output and target to be B x 1 or B x C or target can be B (with ints from 0 to C-1)
    assert output.dim() == 2, "Output should be a 2-dimensional tensor"
    total = target.size(0)
    assert output.size(0) == total, "Outputs and targets should have the same number of samples"
    # Turn it into a 1d class encoding
    if target.dim() == 2:
        if target.size(1) > 1:
            raise NotImplementedError("Multiclass with 2d targets is not impelemented yet")
        target = target.squeeze(1).long()

    # Change the output to have two cols if it only has 1
    if output.size(1) == 1:
        # Want to consider the 0.5 case
        output = (output >= 0.5).float()
        output = torch.cat([1 - output, output], dim=1)

    # Compute the accuracy
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).float().sum(0)
    return (correct / total) * 100.


def accuracy_with_logits(output, target):
    return accuracy(F.sigmoid(output), target)
