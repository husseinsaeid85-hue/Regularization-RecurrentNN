# Regularization-RecurrentNN
This repository contains a flexible recurrent neural network framework that supports various regularization strategies. Regularization is an essential technique in deep learning to prevent overfitting and improve generalization performance.

## Refactoring

To support the training and testing phases, the framework has been extended and refactored. The changes include:

- Addition of a `testing_phase` boolean member in the `BaseLayer` class to indicate the current phase (default: False).
- Implementation of a `phase` property in the `NeuralNetwork` class to set the phase of each layer based on the network's phase (train or test).
- Introduction of a base optimizer class, `Optimizer`, in the `Optimizers.py` file, which serves as the parent class for all optimizers. It provides basic functionality and enables the use of regularizers.

## Optimization Constraints

The repository includes implementation for two common optimization constraints: L2 Regularization and L1 Regularization. These constraints enforce small weights (L2) or sparsity (L1) in the model. Key features include:

- The `L2Regularizer` and `L1Regularizer` classes in the `Constraints.py` file provide methods to calculate gradients and calculate the norm-enhanced loss.
- Optimizers have been refactored to apply the new regularizers using the `calculate_gradient(weights)` method.
- The `NeuralNetwork` class has been updated to include the regularization loss in addition to the data loss. The `norm(weights)` method is used to calculate the regularization loss within all layers (Fully Connected, Convolution, and RNN).

## Dropout Regularization

Dropout is a popular regularization method for fully connected layers. It helps reduce co-adaptation by randomly dropping units during training. Key features include:

- The `Dropout` class in the `Dropout.py` file implements dropout regularization.
- The constructor of the `Dropout` class accepts a `probability` argument to determine the fraction of units to keep.
- The `forward(input_tensor)` and `backward(error_tensor)` methods are implemented for the training phase.

## Batch Normalization

Batch Normalization is a regularization technique specifically designed for deep learning. It helps stabilize and accelerate training by normalizing the inputs within each mini-batch. Key features include:

- The `BatchNormalization` class in the `BatchNormalization.py` file implements batch normalization.
- The constructor of the `BatchNormalization` class accepts a `channels` argument, representing the number of channels in the input tensor.
- The `initialize` method initializes the weights and biases. Weights are set to ones, and biases are set to zeros to avoid impacting training initially.
- The `forward(input_tensor)` and `backward(error_tensor)` methods are implemented with independent activations for the training phase.
- The class also provides a method, `reformat(tensor)`, to reshape the tensor between image-like and vector-like variants.
- A moving average estimation of training set mean and variance is implemented, and an online estimation is used during testing.

## Recurrent Layers

Recurrent Neural Networks (RNNs) are essential for tasks requiring memory and sequential processing, such as time series prediction. This repository includes the implementation of an Elman RNN cell. Key features include:

- Activation functions, TanH and Sigmoid, are implemented in the `TanH.py` and `Sigmoid.py` files in the `Layers` folder.
- The `RNN` class in the `RNN.py` file implements the Elman RNN layer.
- The constructor of the `RNN` class accepts arguments for input size, hidden size, and output size. The hidden state is initialized with zeros.
- The `forward(input_tensor)`method returns a tensor that serves as the input tensor for the next layer.
- The `backward(error_tensor)` method updates the parameters and returns the error tensor for the next layer.
## install the required dependencies:
pip install numpy
from Base import BaseLayer

## Contributing
Contributions to this framework are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue.
