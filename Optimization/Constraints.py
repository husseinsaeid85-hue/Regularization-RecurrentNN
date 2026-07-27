"""Weight constraints (regularizers) that optimizers can apply during updates.

Each regularizer exposes the same two-method interface:

* ``calculate_gradient(weights)`` -- the derivative of the penalty with
  respect to the weights, which an optimizer subtracts from the weights as a
  shrinkage term inside ``calculate_update``.
* ``norm(weights)`` -- the scalar penalty itself, which is added to the data
  loss to obtain the total loss.
"""

import numpy as np


class L2_Regularizer:
    """Ridge penalty ``alpha * ||w||^2``.

    Shrinks every weight proportionally to its own magnitude, which keeps
    weights small and spread out rather than driving any of them to zero.
    """

    def __init__(self, alpha):
        """
        :param alpha: regularization strength
        """
        self.regular_weight = alpha

    def calculate_gradient(self, weights):
        """
        :param weights: weight tensor
        :return: alpha * w, the sub-gradient of the L2 penalty
        """
        return self.regular_weight * weights

    def norm(self, weights):
        """
        :param weights: weight tensor
        :return: the scalar L2 penalty, alpha * ||w^2||2
        """
        l2_norm_term = self.regular_weight * np.sum(weights ** 2)
        return l2_norm_term


class L1_Regularizer:
    """Lasso penalty ``alpha * ||w||``.

    Shrinks every weight by a constant amount regardless of its magnitude,
    which pushes small weights all the way to zero and yields sparse weights.
    """

    def __init__(self, alpha):
        """
        :param alpha: regularization strength
        """
        self.regular_weight = alpha

    def calculate_gradient(self, weights):
        """
        :param weights: weight tensor
        :return: alpha * sign(w), the sub-gradient of the L1 penalty
        """
        return self.regular_weight * np.sign(weights)

    def norm(self, weights):
        """
        :param weights: weight tensor
        :return: the scalar L1 penalty, alpha * ||w||
        """
        l1_norm_term = self.regular_weight * np.sum(np.abs(weights))
        return l1_norm_term
