"""Gradient descent optimizers, all sharing a regularizer-aware base class."""

import math

import numpy as np


class Optimizer:
    """Base class for every optimizer.

    Holds the optional regularizer. Subclasses implement
    ``calculate_update(weight_tensor, gradient_tensor)`` and, when a
    regularizer is attached, apply its shrinkage term to the weights before
    applying the data gradient.
    """

    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        """Attach an L1 or L2 regularizer to this optimizer.

        :param regularizer: object exposing ``calculate_gradient`` and ``norm``
        """
        self.regularizer = regularizer


class Sgd(Optimizer):
    """Plain stochastic gradient descent."""

    def __init__(self, learning_rate: float):
        """
        :param learning_rate: step size
        """
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        :param weight_tensor: current weights
        :param gradient_tensor: gradient of the data loss w.r.t. the weights
        :return: updated weights
        """
        if self.regularizer is None:
            new_weight_tensor = weight_tensor - (self.learning_rate * gradient_tensor)
        else:
            shrink = weight_tensor - (self.learning_rate * self.regularizer.calculate_gradient(weight_tensor))
            new_weight_tensor = shrink - (self.learning_rate * gradient_tensor)

        return new_weight_tensor


class SgdWithMomentum(Optimizer):
    """SGD with a momentum term that smooths the update direction."""

    def __init__(self, learning_rate, momentum_rate):
        """
        :param learning_rate: step size
        :param momentum_rate: fraction of the previous update carried forward
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.momentum_term = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        :param weight_tensor: current weights
        :param gradient_tensor: gradient of the data loss w.r.t. the weights
        :return: updated weights
        """
        self.momentum_term = (self.momentum_rate * self.momentum_term) - (self.learning_rate * gradient_tensor)
        if self.regularizer is None:
            weight_tensor = weight_tensor + self.momentum_term
        else:
            shrink = weight_tensor - (self.learning_rate * self.regularizer.calculate_gradient(weight_tensor))
            weight_tensor = shrink + self.momentum_term
        return weight_tensor


class Adam(Optimizer):
    """Adam: momentum on the gradient and on its element-wise square."""

    def __init__(self, lr, mu, rho):
        """
        :param lr: step size
        :param mu: decay rate of the first moment estimate
        :param rho: decay rate of the second moment estimate
        """
        super().__init__()
        self.learning_rate = lr
        self.mu = mu
        self.rho = rho
        self.v_term = 0
        self.r_term = 0
        self.it = 1
        self.epsilon = np.finfo(float).eps

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        :param weight_tensor: current weights
        :param gradient_tensor: gradient of the data loss w.r.t. the weights
        :return: updated weights
        """
        self.v_term = (self.mu * self.v_term) + ((1 - self.mu) * gradient_tensor)
        self.r_term = self.rho * self.r_term + (1 - self.rho) * np.multiply(gradient_tensor, gradient_tensor)

        # bias correction, needed because both moments start at zero
        v_hat = self.v_term / (1 - math.pow(self.mu, self.it))
        r_hat = self.r_term / (1 - math.pow(self.rho, self.it))

        adam_term = v_hat / (np.sqrt(r_hat) + self.epsilon)

        if self.regularizer is None:
            weight_tensor = weight_tensor - self.learning_rate * adam_term

        else:
            shrink = weight_tensor - (self.learning_rate * self.regularizer.calculate_gradient(weight_tensor))
            weight_tensor = shrink - self.learning_rate * adam_term

        self.it = self.it + 1

        return weight_tensor
