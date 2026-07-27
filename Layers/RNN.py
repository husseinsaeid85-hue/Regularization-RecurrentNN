"""Elman recurrent layer, built from two fully connected layers."""

import numpy as np

from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.Sigmoid import Sigmoid
from Layers.TanH import TanH


class RNN(BaseLayer):
    """Elman RNN cell unrolled over the batch dimension as time.

    Each row of the input tensor is treated as one time step. Per step the
    previous hidden state is concatenated with the current input and pushed
    through a fully connected layer plus tanh to produce the new hidden
    state; a second fully connected layer plus sigmoid maps that hidden state
    to the output.

    ``memorize`` controls whether the hidden state carries over between
    consecutive ``forward`` calls (a truncated-BPTT style long sequence) or is
    reset to zeros at the start of every call (independent sequences).
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        :param input_size: number of features per time step
        :param hidden_size: dimensionality of the hidden state
        :param output_size: number of features produced per time step
        """
        # BaseLayer.__init__ is not called here: it assigns ``self.weights``,
        # which this class exposes as a property backed by the hidden fully
        # connected layer. That layer does not exist yet at this point, and
        # assigning it afterwards would wipe the initialized weights.
        self.trainable = True
        self.testing_phase = False
        self.regular_loss = 0.0

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(hidden_size)

        self._memorize = False
        self._optimizer = None
        self._gradient_weights = None

        self.tanh = TanH()
        self.sigmoid = Sigmoid()

        self.ful_con_layer_hidden = FullyConnected(self.hidden_size + self.input_size, self.hidden_size)
        self.ful_con_layer_output = FullyConnected(self.hidden_size, self.output_size)

        # per-time-step caches, refilled on every forward pass
        self.sigmoid_activations = []
        self.tanh_activations = []
        self.output_ful_con_input_tensors = []
        self.ful_con_layer_hiddens_input_tensors = []
        self.ful_con_layer_hidden_gradient_weights = []
        self.ful_con_layer_output_gradient_weights = None

    def forward(self, input_tensor):
        """Unroll the cell over the batch dimension.

        :param input_tensor: dim= [time_steps, input_size]
        :return: dim= [time_steps, output_size]
        """
        self.sigmoid_activations = []
        self.tanh_activations = []
        self.output_ful_con_input_tensors = []
        self.ful_con_layer_hiddens_input_tensors = []
        self.ful_con_layer_hidden_gradient_weights = []
        if self.memorize:
            last_hidden_layer = self.hidden_state
        else:
            last_hidden_layer = np.zeros(self.hidden_size)

        batch = input_tensor.shape[0]
        output_tensor = np.ndarray((batch, self.output_size))

        for b in range(batch):
            x_t = np.concatenate([last_hidden_layer, input_tensor[b]]).reshape(1, -1)
            tanh_tensor = self.ful_con_layer_hidden.forward(x_t)
            current_hidden_state = self.tanh.forward(tanh_tensor)

            # update last hidden layer
            last_hidden_layer = current_hidden_state[0]

            # transition of hy
            sigmoid_tensor = self.ful_con_layer_output.forward(current_hidden_state)
            sigmoid_output_tensor = self.sigmoid.forward(sigmoid_tensor)

            output_tensor[b] = sigmoid_output_tensor[0]

            # cache everything the backward pass needs to replay this time step
            self.ful_con_layer_hiddens_input_tensors.append(self.ful_con_layer_hidden.input_tensor)

            self.output_ful_con_input_tensors.append(self.ful_con_layer_output.input_tensor)
            self.sigmoid_activations.append(self.sigmoid.activations)
            self.tanh_activations.append(self.tanh.activations)

            # update hidden state
            self.hidden_state = current_hidden_state[0]

        return output_tensor

    def backward(self, error_tensor):
        """Backpropagate through time, from the last time step to the first.

        :param error_tensor: dim= [time_steps, output_size]
        :return: gradient with respect to the input, dim= [time_steps, input_size]
        """
        self.gradient_weights = np.zeros_like(self.ful_con_layer_hidden.weights)
        self.ful_con_layer_output_gradient_weights = np.zeros_like(self.ful_con_layer_output.weights)
        grad_last_hid_layer = 0
        batch = error_tensor.shape[0]
        gradient_inputs = np.zeros((batch, self.input_size))

        time_step = batch - 1

        while time_step >= 0:

            # output
            self.sigmoid.activations = self.sigmoid_activations[time_step]
            self.ful_con_layer_output.input_tensor = self.output_ful_con_input_tensors[time_step]
            # use sigmoid error as input
            ful_con_layer_output_error = self.ful_con_layer_output.backward(
                self.sigmoid.backward(error_tensor[time_step]))

            # hidden
            self.tanh.activations = self.tanh_activations[time_step]
            self.ful_con_layer_hidden.input_tensor = self.ful_con_layer_hiddens_input_tensors[time_step]
            # use tanh error as input; the hidden branch also receives the
            # gradient flowing back from the following time step
            ful_con_layer_hidden_error = self.ful_con_layer_hidden.backward(
                self.tanh.backward(ful_con_layer_output_error + grad_last_hid_layer))

            # gradient last hidden layer
            grad_last_hid_layer = ful_con_layer_hidden_error[:, :self.hidden_size]

            # gradient with respect to input
            gradient_inputs[time_step] = ful_con_layer_hidden_error[:, self.hidden_size:][0]

            # accumulate the weight gradients across all time steps
            self.gradient_weights += self.ful_con_layer_hidden.gradient_weights
            self.ful_con_layer_output_gradient_weights += self.ful_con_layer_output.gradient_weights

            # update time step
            time_step -= 1

        if self.optimizer:
            self.ful_con_layer_output.weights = self.optimizer.calculate_update(
                self.ful_con_layer_output.weights, self.ful_con_layer_output_gradient_weights)
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)  # hidden

        return gradient_inputs

    def calculate_regularization_loss(self):
        """Accumulate the regularizer norm over both internal weight matrices."""
        if self.optimizer.regularizer:
            self.regular_loss += (self.optimizer.regularizer.norm(self.ful_con_layer_hidden.weights)
                                  + self.optimizer.regularizer.norm(self.ful_con_layer_output.weights))
        return self.regular_loss

    def initialize(self, weights_initializer, bias_initializer):
        """Initialize both internal fully connected layers."""
        self.ful_con_layer_hidden.initialize(weights_initializer, bias_initializer)
        self.ful_con_layer_output.initialize(weights_initializer, bias_initializer)

    @property
    def memorize(self):
        """Whether the hidden state carries over between forward calls."""
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def weights(self):
        """The hidden transition weights, owned by the inner FullyConnected layer."""
        return self.ful_con_layer_hidden.weights

    @weights.setter
    def weights(self, value):
        self.ful_con_layer_hidden.weights = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value
