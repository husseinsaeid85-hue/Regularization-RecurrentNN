"""Base class shared by every layer in the framework."""


class BaseLayer:
    """Common interface and state for all layers.

    Attributes:
        trainable: True if the layer owns weights that an optimizer updates.
        weights: The layer's weight tensor, or None for parameter-free layers.
        testing_phase: False during training, True during inference. The
            NeuralNetwork sets this on every layer through its ``phase``
            property so that layers such as Dropout and BatchNormalization
            can switch behaviour between training and testing.
    """

    def __init__(self):
        self.trainable = False
        self.weights = None
        self.testing_phase = False
