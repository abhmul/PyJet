import torch.nn as nn
import torch.nn.functional as F

from .. import backend as J
from ..utils import deprecated

pool_types = {
    "no_pool": lambda *args, **kwargs: lambda x: x,
    "max": nn.MaxPool1d,
    "avg": nn.AvgPool1d,
}
activation_types = {
    name.lower(): cls
    for name, cls in nn.modules.activation.__dict__.items()
    if isinstance(cls, type)
}
activation_types["linear"] = None


def get_type(item_type, type_dict, fail_message):
    try:
        if not isinstance(item_type, str):
            return item_type
        return type_dict[item_type]
    except KeyError:
        raise NotImplementedError(fail_message)


def get_pool_type(pool_type):
    return get_type(pool_type, pool_types, "pool type %s" % pool_type)


def get_activation_type(activation_type):
    return get_type(
        activation_type, activation_types, "Activation %s" % activation_type
    )


def construct_n_layers(
    layer_factory, num_layers, input_size, output_size, *args, **kwargs
):
    layers = nn.ModuleList([layer_factory(input_size, output_size, *args, **kwargs)])
    for _ in range(num_layers - 1):
        layers.append(layer_factory(output_size, output_size, *args, **kwargs))
    return layers


def get_input_shape(inputs):
    """Gets the shape of the input excluding the batch size"""
    return tuple(inputs.size())[1:]


def get_shape_no_channels(inputs, channels_mode=J.channels_mode):
    """Gets the shape of the input excluding the batch size and channels"""
    no_batch_size = get_input_shape(inputs)
    if channels_mode == "channels_first":
        return no_batch_size[1:]
    else:
        return no_batch_size[:-1]


def get_channels(inputs, channels_mode=J.channels_mode):
    input_shape = get_input_shape(inputs)
    if channels_mode == "channels_first":
        return input_shape[0]
    else:
        return input_shape[-1]


@deprecated
def builder(func):
    def build_layer(self, *args, **kwargs):
        assert not self.built, "Cannot build a layer multiple times!"
        func(self, *args, **kwargs)
        if J.use_cuda:
            self.cuda()
        self.built = True

    return build_layer
