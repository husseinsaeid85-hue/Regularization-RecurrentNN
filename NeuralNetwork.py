"""Container that wires layers together and drives training and testing."""

import copy

import numpy as np

from Layers.Base import BaseLayer


class NeuralNetwork(BaseLayer):
    """A sequential stack of layers plus the machinery to train it.

    Layers are appended in order. Every trainable layer gets its own deep copy
    of the optimizer, so stateful optimizers such as Adam and SgdWithMomentum
    keep per-layer moments instead of sharing one set across the network.

    The ``phase`` property is the single switch between training and
    inference: setting it propagates the phase down to every layer, which is
    what lets Dropout stop dropping and BatchNormalization swap batch
    statistics for its moving average.
    """

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        """
        :param optimizer: optimizer instance, deep-copied per trainable layer
        :param weights_initializer: initializer for layer weight tensors
        :param bias_initializer: initializer for layer bias tensors
        """
        super().__init__()

        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_buffer = None
        self._phase = None

    @property
    def phase(self):
        """Either ``'train'`` or ``'test'``."""
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value
        for layer in self.layers:
            layer.phase = value
            layer.testing_phase = (value == 'test')

    def forward(self):
        """Pull one batch from the data layer and run it through to the loss.

        :return: the scalar data loss for this batch
        """
        input_tensor, self.label_buffer = self.data_layer.next()

        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        output = self.loss_layer.forward(input_tensor, self.label_buffer)

        return output

    def backward(self):
        """Propagate the error from the loss layer back through every layer."""
        error_tensor = self.loss_layer.backward(self.label_buffer)

        for layer in np.flip(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        """Add a layer to the stack.

        Trainable layers are initialized with the network-wide initializers
        and receive their own deep copy of the optimizer.

        :param layer: a BaseLayer subclass instance
        """
        if layer.trainable:
            layer.initialize(self.weights_initializer, self.bias_initializer)
            deep_copy = copy.deepcopy(self.optimizer)
            layer.optimizer = deep_copy

        self.layers.append(layer)

    def train(self, iterations):
        """Run ``iterations`` forward/backward passes, recording the loss.

        :param iterations: number of training steps
        """
        self.phase = 'train'
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        """Run a forward pass in inference mode.

        :param input_tensor: input batch
        :return: the network's output for that batch
        """
        self.phase = 'test'
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        results = input_tensor

        return results
