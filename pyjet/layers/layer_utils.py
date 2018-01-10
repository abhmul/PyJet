import torch.nn as nn
import torch.nn.functional as F

pool_types = {"no_pool": lambda *args, **kwargs: lambda x: x,
              "max": nn.MaxPool1d,
              "avg": nn.AvgPool1d}
activation_types = {"linear": None,
                    "relu": nn.ReLU,
                    "softmax": nn.Softmax,
                    "tanh": nn.Tanh}


def get_type(item_type, type_dict, fail_message):
    try:
        return type_dict[item_type]
    except KeyError:
        raise NotImplementedError(fail_message)


def get_pool_type(pool_type):
    return get_type(pool_type, pool_types, "pool type %s" % pool_type)


def get_activation_type(activation_type):
    return get_type(activation_type, activation_types, "Activation %s" % activation_type)


def construct_n_layers(layer_factory, num_layers, input_size, output_size, *args, **kwargs):
    layers = nn.ModuleList([layer_factory(input_size, output_size, *args, **kwargs)])
    for _ in range(num_layers - 1):
        layers.append(layer_factory(output_size, output_size, *args, **kwargs))
    return layers
