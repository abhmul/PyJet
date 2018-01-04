import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import backend as J
flatten = J.flatten

# TODO: Add a cropping function


def pad_tensor(tensor, length, pad_value=0.0, dim=0):
    # tensor is Li x E
    tensor = tensor.transpose(0, dim).contiguous()
    if tensor.size(0) == length:
        tensor = tensor
    elif tensor.size(0) > length:
        return tensor[:length]
    else:
        tensor = torch.cat([tensor, Variable(J.zeros(length - tensor.size(0), *tensor.size()[1:]).fill_(pad_value),
                                             requires_grad=False)])
    return tensor.transpose(0, dim).contiguous()


def pad_sequences(tensors, pad_value=0.0, length_last=False):
    # tensors is B x Li x E
    # First find how long we need to pad until
    length_dim = -1 if length_last else 0
    assert len(tensors) > 0
    if length_last:
        assert all(tuple(seq.size())[:-1] == tuple(tensors[0].size())[:-1] for seq in tensors)
    else:
        assert all(tuple(seq.size())[1:] == tuple(tensors[0].size())[1:] for seq in tensors)
    seq_lens = [seq.size(length_dim) for seq in tensors]
    max_len = max(seq_lens)
    # Out is B x L x E
    # print([tuple(pad_tensor(tensors[i], max_len).size()) for i in range(len(tensors))])
    if length_last:
        return torch.stack(
            [pad_tensor(tensors[i].transpose(0, length_dim), max_len, pad_value=pad_value).transpose(0, length_dim)
             for i in range(len(tensors))]), seq_lens

    return torch.stack([pad_tensor(tensors[i], max_len, pad_value=pad_value) for i in range(len(tensors))]), seq_lens


def unpad_sequences(padded_tensors, seq_lens, length_last=False):
    length_dim = -1 if length_last else 0
    if length_last:
        return [padded_tensor.transpose(0, length_dim)[:seq_len].transpose(0, length_dim) for padded_tensor, seq_len in
                zip(padded_tensors, seq_lens)]
    return [padded_tensor[:seq_len] for padded_tensor, seq_len in zip(padded_tensors, seq_lens)]


def pack_sequences(tensors):
    # tensors is B x Li x E
    assert len(tensors) > 0
    assert all(seq.size(1) == tensors[0].size(1) for seq in tensors)
    seq_lens = [seq.size(0) for seq in tensors]
    return torch.cat(tensors), seq_lens


def unpack_sequences(packed_tensors, seq_lens):
    # Find the start inds of all of the sequences
    seq_starts = [0 for _ in range(len(seq_lens))]
    seq_starts[1:] = [seq_starts[i-1] + seq_lens[i-1] for i in range(1, len(seq_starts))]
    # Unpack the tensors
    return [packed_tensors[seq_starts[i]:seq_starts[i] + seq_lens[i]] for i in range(len(seq_lens))]


def kmax_pooling(x, dim, k):
    index = x.topk(min(x.size(dim), k), dim=dim)[1].sort(dim=dim)[0]
    x = x.gather(dim, index)
    if x.size(dim) < k:
        x = pad_tensor(x, k, dim=dim)
    return x


def pad_numpy_to_length(x, length):
    if len(x) < length:
        return np.concatenate([x, np.zeros((length - len(x),) + x.shape[1:])], axis=0)
    return x


def seq_softmax(x, return_padded=False):
    # x comes in as B x Li x F, we compute the softmax over Li for each F
    x, lens = pad_sequences(x, pad_value=-float('inf'))  # B x L x F
    shape = tuple(x.size())
    assert len(shape) == 3
    x = F.softmax(x, dim=1)
    assert tuple(x.size()) == shape
    if return_padded:
        return x, lens
    # Un-pad the tensor and return
    return unpad_sequences(x, lens)  # B x Li x F
