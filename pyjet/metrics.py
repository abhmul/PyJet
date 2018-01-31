import torch

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
    return topk_accuracy(output, target, topk=1)
