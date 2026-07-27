"""Hyperbolic tangent activation, used as the RNN hidden-state non-linearity."""

import numpy as np

from Layers.Base import BaseLayer


class TanH(BaseLayer):
    """Element-wise tanh activation.

    The forward pass caches its own output in ``activations`` because the
    derivative of tanh can be expressed purely in terms of that output:
    ``d/dx tanh(x) = 1 - tanh(x) ** 2``. The RNN layer relies on this cache
    being writable so it can restore the activation of a specific time step
    while backpropagating through time.
    """

    def __init__(self):
        super().__init__()
        self.activations = 0

    def forward(self, input_tensor):
        """Apply tanh element-wise.

        :param input_tensor: input of arbitrary shape
        :return: tanh(input_tensor), also cached in ``self.activations``
        """
        self.activations = np.tanh(input_tensor)
        return self.activations

    def backward(self, error_tensor):
        """Propagate the error through the tanh non-linearity.

        :param error_tensor: gradient with respect to this layer's output
        :return: gradient with respect to this layer's input
        """
        error_tensor = error_tensor * (1 - np.square(self.activations))
        return error_tensor
