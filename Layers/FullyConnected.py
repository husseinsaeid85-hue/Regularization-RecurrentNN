"""Fully connected (affine) layer."""

import numpy as np

from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    """Trainable affine layer: ``y = [x, 1] @ W``.

    The bias is folded into the weight matrix as an extra input row, so the
    forward pass appends a constant 1 to every sample and a single matmul
    covers both the linear term and the bias.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        :param input_size: number of input features (excluding the bias)
        :param output_size: number of output features
        """
        super().__init__()

        self.input_size = input_size + 1  # adding bias
        self.output_size = output_size

        self.trainable = True
        self.weights = np.random.rand(self.input_size, self.output_size)

        self.input_buffer = None

        self._optimizer = None
        self._gradient_weights = None

    def initialize(self, weights_initializer, bias_initializer):
        """Re-draw the weights and the bias row from the given initializers.

        :param weights_initializer: object exposing ``initialize(shape, fan_in, fan_out)``
        :param bias_initializer: object exposing ``initialize(shape, fan_in, fan_out)``
        """
        weights = weights_initializer.initialize((self.input_size - 1, self.output_size),
                                                 fan_in=self.input_size - 1,
                                                 fan_out=self.output_size)

        bias = bias_initializer.initialize((1, self.output_size),
                                           fan_in=1,
                                           fan_out=self.output_size)

        self.weights = np.concatenate((weights, bias))

    def forward(self, input_tensor):
        """
        :param input_tensor: dim= [b:batch_size, n:input_size]
        :return: dim= [b, output_size]
        """
        input_tensor = np.append(input_tensor, np.ones((input_tensor.shape[0], 1)), axis=1)
        self.input_buffer = input_tensor

        next_input_tensor = np.matmul(input_tensor, self.weights)

        return next_input_tensor

    def backward(self, error_tensor):
        """Compute the weight gradient, update the weights, and pass the error on.

        :param error_tensor: dim= [b, output_size]
        :return: gradient with respect to the input, dim= [b, input_size]
        """
        prev_error_tensor = np.matmul(error_tensor, np.transpose(self.weights))
        # drop the column belonging to the appended bias input
        prev_error_tensor = np.delete(prev_error_tensor, prev_error_tensor.shape[1] - 1, axis=1)

        gradient_tensor = np.matmul(np.transpose(self.input_buffer), error_tensor)
        self.gradient_weights = gradient_tensor

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, gradient_tensor)

        return prev_error_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
