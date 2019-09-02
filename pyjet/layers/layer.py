import torch.nn as nn


class Layer(nn.Module):
    def __init__(self):
        """An abstract class for a layer. This subclasses `torch`'s `nn.Module`.

        Implement this to get a PyJet specific layer. You can include a builder
        for the function if it cannot be built upon instantiation. Simply write
        a function that modifies your layer to have all necessary parameters
        instantiated and annotate it with the `@builder` decorator from Pyjet's
        `layers` package. Then you can call it at the top of your `forward` and
        it will only run once.
        """
        super(Layer, self).__init__()
        self.built = False

    def forward(self, *input):
        """Implement this to define the input to output logic of the layer."""
        raise NotImplementedError()

    def reset_parameters(self):
        """Implement this to make this layer's parameters resettable."""
        pass

    def weight(self, *args, **kwargs):
        """Implement this to query your model for its specific weights"""
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__ + "()"

    def __repr__(self):
        return str(self)
