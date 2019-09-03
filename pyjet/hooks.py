import logging
import torch
import torch.nn as nn
from pyjet.layers import Layer, Input
from pyjet.layers.layer_utils import get_input_shape
from . import utils


class Hook:
    def __init__(
        self,
        module: nn.Module,
        inner_hook_func,
        is_forward=True,
        pre_hook=False,
        detach=True,
    ):
        """Creates a hook on `module` with hook function `inner_hook_func`. The `inner_hook_func`
        takes as input
        - `module`
        - `inputs`: the inputs to the module in a list
        - `outputs`: the outputs of the module in a list
        
        Arguments:
            module {nn.Module} -- The module to hook
            inner_hook_func -- The function to run in the hook
        
        Keyword Arguments:
            is_forward {bool} -- Creates a forward hook. If False, creates a backward hook instead  (default: {True})
            pre_hook {bool} -- Creates a forward hook that runs before the forward call. `is_forward` must be true (default: {False})
            detach {bool} -- Detaches the inputs and outputs from the computation graph  (default: {True})
        """
        self.inner_hook_func = inner_hook_func
        self.is_forward = is_forward
        self.pre_hook = pre_hook
        self.detach = detach
        self.stored = None

        # Access the map with (is_forward, pre_hook)
        hook_type_map = {
            (True, True): module.register_forward_pre_hook,
            (True, False): module.register_forward_hook,
            (False, False): module.register_backward_hook,
        }
        hooker = hook_type_map[(is_forward, pre_hook)]

        self.hook = hooker(
            self._hook_func_pre_hook if self.pre_hook else self._hook_func_post_hook
        )
        self.removed = False
        logging.info(f"Hooked function {inner_hook_func.__name__} onto module {module}")

    def _hook_func_post_hook(self, module: nn.Module, inputs, outputs):
        """Applies `hook_func` to `module`, `inputs`, `outputs`."""

        inputs = utils.standardize_list_input(inputs)
        outputs = utils.standardize_list_input(outputs)
        if self.detach:
            inputs = [i.detach() for i in inputs]
            outputs = [o.detach() for o in outputs]
        self.stored = self.inner_hook_func(module, inputs, outputs)

    def _hook_func_pre_hook(self, module: nn.Module, inputs):
        """Applies `hook_func` to `module`, `inputs`."""
        inputs = utils.standardize_list_input(inputs)
        if self.detach:
            inputs = [i.detach() for i in inputs]
        self.stored = self.inner_hook_func(module, inputs)

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


class Hooks:
    "Create several hooks on the modules in `modules` with `inner_hook_func`."

    def __init__(
        self, modules, inner_hook_func, is_forward=True, pre_hook=False, detach=True
    ):
        self.hooks = [
            Hook(module, inner_hook_func, is_forward, pre_hook, detach)
            for module in modules
        ]

    def __getitem__(self, i: int) -> Hook:
        return self.hooks[i]

    def __len__(self) -> int:
        return len(self.hooks)

    def __iter__(self):
        return iter(self.hooks)

    @property
    def stored(self):
        return [hook.stored for hook in self]

    def remove(self):
        "Remove the hooks from the model."
        for h in self.hooks:
            h.remove()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


def _hook_inner(module, inputs, outputs):
    return outputs


def hook_output(module: nn.Module, detach=True, grad=False) -> Hook:
    """Returns a `Hook` that stores the activations of `module` in `self.stored`

    Arguments:
        module {nn.Module} -- The module to hook
    
    Keyword Arguments:
        detach {bool} -- Detaches the inputs and outputs from the computation graph  (default: {True})
        grad {bool} -- Gets the gradient output instead (basically a backward hook)  (default: {False})
    """
    return Hook(module, _hook_inner, detach=detach, is_forward=not grad)


def hook_outputs(modules, detach: bool = True, grad: bool = False) -> Hooks:
    "Return `Hooks` that store activations of all `modules` in `self.stored`"
    return Hooks(modules, _hook_inner, detach=detach, is_forward=not grad)


def model_sizes(model: nn.Module, input_tensor):
    """
    Pass a dummy input through the model `model` to get the various sizes of activations.
    Requires that the child modules of `model` return only a single tensor output.
    """

    def process_model_output(output):
        output = utils.destandardize_list_input(output)
        return get_input_shape(output)

    with hook_outputs(model.children()) as hooks:
        model(input_tensor)

        return [process_model_output(o.stored) for o in hooks]
