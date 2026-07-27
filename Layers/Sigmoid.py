"""Logistic sigmoid activation, used on the RNN output projection."""

import numpy as np

from Layers.Base import BaseLayer


class Sigmoid(BaseLayer):
    """Element-wise logistic sigmoid activation.

    As with TanH, the forward pass caches its output in ``activations``
    because the derivative is a function of the output alone:
    ``d/dx sigma(x) = sigma(x) * (1 - sigma(x))``. The RNN layer overwrites
    this cache per time step during backpropagation through time.
    """

    def __init__(self):
        super().__init__()
        self.activations = 0

    def forward(self, input_tensor):
        """Apply the logistic sigmoid element-wise.

        :param input_tensor: input of arbitrary shape
        :return: sigma(input_tensor), also cached in ``self.activations``
        """
        self.activations = 1 / (1 + np.exp(-input_tensor))
        return self.activations

    def backward(self, error_tensor):
        """Propagate the error through the sigmoid non-linearity.

        :param error_tensor: gradient with respect to this layer's output
        :return: gradient with respect to this layer's input
        """
        error_tensor = error_tensor * self.activations * (1 - self.activations)
        return error_tensor
