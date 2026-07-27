"""Inverted dropout regularization for fully connected layers."""

import numpy as np

from Layers.Base import BaseLayer


class Dropout(BaseLayer):
    """Randomly zeroes activations during training to reduce co-adaptation.

    This is *inverted* dropout: surviving activations are rescaled by
    ``1 / probability`` during training, so the expected activation is
    unchanged and the testing phase becomes a plain identity mapping with no
    rescaling of its own.
    """

    def __init__(self, probability):
        """
        :param probability: fraction of units to KEEP, in (0, 1]
        """
        super().__init__()
        self.probability = probability
        self._drop_mask = None

    def forward(self, input_tensor):
        """
        :param input_tensor: input dim: [b:batch_size, n:input_size]
        :return: dropped out input_tensor with given probability
        """
        if self.testing_phase is False:
            # create a random array for removing neurons
            self._drop_mask = np.random.uniform(0, 1, (input_tensor.shape[0], input_tensor.shape[1]))
            # if val < probability -> drop_mask = 1 (keep), else: 0 (drop)
            self._drop_mask = np.where(self._drop_mask < self.probability, 1, 0)
            input_tensor = input_tensor * self._drop_mask  # keep neurons with given probability
            input_tensor = input_tensor * (1 / self.probability)
        return input_tensor

    def backward(self, error_tensor):
        """
        :param error_tensor: with dim: [b, m]
        :return: error_tensor which shut down neurons whose got 0 value in forward pass
        """
        error_tensor = self._drop_mask * error_tensor / self.probability
        return error_tensor
