"""Batch normalization for both vector-like and image-like activations."""

import copy

import numpy as np

from Layers import Helpers
from Layers.Base import BaseLayer


class BatchNormalization(BaseLayer):
    """Normalizes activations per channel, then rescales with learned gamma/beta.

    During training the batch mean and variance are used directly, and a
    moving average of both is accumulated. During testing that moving average
    stands in for the batch statistics, so inference does not depend on the
    other samples that happen to share a batch.

    Image-like inputs of shape [b, c, x, y] are folded down to [b * x * y, c]
    by ``reformat`` so that the same per-channel statistics apply to every
    spatial position, and folded back before the result is returned.
    """

    def __init__(self, channel):
        """
        :param channel: number of channels in the input tensor
        """
        super().__init__()
        self.channel = channel
        self.trainable = True
        self.input_mean = 0
        self.input_var = 0
        self.move_avg_mean = 0
        self.move_avg_var = 0
        self.alpha = 0.8  # momentum of the moving average
        self.epsilon = 1e-13
        self.iteration = 0
        self.tensor_shape = None
        self._last_input = None
        self._optimizer = None
        self._bias_optimizer = None
        self._gradient_weights = None
        self._gradient_bias = None
        self.normalized_input = None

    def initialize(self, weights_initializer=None, bias_initializer=None):
        """Set gamma to ones and beta to zeros.

        Batch normalization deliberately ignores the network-wide
        initializers: starting from an identity transform keeps the layer from
        distorting the signal before it has learned anything. The arguments
        exist only so the layer matches the ``initialize`` signature that
        ``NeuralNetwork.append_layer`` calls on every trainable layer.
        """
        self.weights = np.ones(self.channel)
        self.bias = np.zeros(self.channel)

    def reformat(self, tensor):
        """
        :param tensor: dim=[b, c, x, y] or [b * x * y, c]
        :return: img2vec_dim= [b * x * y, c], vec2img_dim= [b, c, x, y]
        """
        if len(tensor.shape) == 4:  # img2vec (4D -> 2D)
            self.tensor_shape = tensor.shape  # store dimension of 4d vector
            # tensor shape = [B × H × M × N]
            # reshape the B × H × M × N tensor to B × H × M·N
            tensor = np.reshape(tensor, (tensor.shape[0], tensor.shape[1], tensor.shape[2] * tensor.shape[3]))
            # transpose from B × H × M·N to B × M·N × H
            tensor = np.swapaxes(tensor, 1, 2)
            # reshape again to have a B.M.N × H tensor
            tensor = np.reshape(tensor, (tensor.shape[0] * tensor.shape[1], tensor.shape[2]))
        else:
            # B * M.N * H
            tensor = np.reshape(tensor,
                                (self.tensor_shape[0], self.tensor_shape[2] * self.tensor_shape[3], tensor.shape[1]))
            # B * H * M.N
            tensor = np.swapaxes(tensor, 2, 1)
            # B * H * M * N
            tensor = np.reshape(tensor, (self.tensor_shape[0], self.tensor_shape[1],
                                         self.tensor_shape[2], self.tensor_shape[3]))

        return tensor

    def forward(self, input_tensor):
        """
        :param input_tensor: dim= [b, c, x, y]
        :return: output_tensor: dim= [b, c, x, y]
        """
        self._last_input = input_tensor
        if len(input_tensor.shape) == 4:  # convert 4D image(vec) to 2D vector
            self._last_input = self.reformat(input_tensor)

        if self.testing_phase is False:  # train phase
            self.input_mean = np.mean(self._last_input, axis=0)
            self.input_var = np.var(self._last_input, axis=0)
            self.normalized_input = (self._last_input - self.input_mean) / np.sqrt(self.input_var + self.epsilon)
            output = self.weights * self.normalized_input + self.bias  # gamma * norm_x + beta

            #  we have to reverse this before returning the output
            if len(input_tensor.shape) == 4:  # convert 2D vector to 4D image(vec)
                output = self.reformat(output)

            if self.iteration == 0:
                # Initialize mean and variance
                # with the batch mean and the batch standard deviation of the first batch used for training
                self.move_avg_mean = self.input_mean
                self.move_avg_var = self.input_var
                self.iteration += 1
            else:
                self.move_avg_mean = self.alpha * self.move_avg_mean + (1 - self.alpha) * self.input_mean
                self.move_avg_var = self.alpha * self.move_avg_var + (1 - self.alpha) * self.input_var

        else:  # test phase
            output = (self._last_input - self.move_avg_mean) / np.sqrt(self.move_avg_var + self.epsilon)
            output = self.weights * output + self.bias

            #  we have to reverse this before returning the output
            if len(input_tensor.shape) == 4:  # convert 2D vector to 4D image(vec)
                output = self.reformat(output)

        return output

    def backward(self, error_tensor):
        """
        :param error_tensor: [b , n] for vector input or [b, c, x, y] for image input
        :return: gradient with respect to the input, in the shape it arrived in
        """
        # we should return the error tensor and compute weight gradient and bias gradient
        error_tensor_shape = error_tensor.shape
        if len(error_tensor_shape) == 4:
            error_tensor = self.reformat(error_tensor)

        output_tensor = Helpers.compute_bn_gradients(error_tensor, self._last_input,
                                                     self.weights, self.input_mean, self.input_var)

        if len(error_tensor_shape) == 4:
            output_tensor = self.reformat(output_tensor)

        self._gradient_weights = np.sum((self.normalized_input * error_tensor), axis=0)  # dl/ dw
        self._gradient_bias = np.sum(error_tensor, axis=0)  # dl/db

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer.calculate_update(self.bias, self._gradient_bias)

        return output_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, val):
        self._optimizer = val
        self._bias_optimizer = copy.deepcopy(val)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, val):
        self._gradient_weights = val

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, val):
        self._gradient_bias = val
