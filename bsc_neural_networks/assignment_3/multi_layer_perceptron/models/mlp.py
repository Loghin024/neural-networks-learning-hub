import numpy as np
from typing import List
from utils.utils import *


class MLP:
    def __init__(self, layer_sizes: List[int], activation_functions: List[str], initial_learning_rate=0.01, patience=5,
                 decay_factor=0.5):
        self.z_values = None
        self.activations = None
        self.num_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []
        self.learning_rate = initial_learning_rate
        self.patience = patience
        self.decay_factor = decay_factor

        # initialize weights and biases for each layer
        for i in range(self.num_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            if i < len(activation_functions) and activation_functions[i] == "relu":
                self.weights.append(he_initialization(fan_in, fan_out))
            elif i < len(activation_functions) and activation_functions[i] == "sigmoid":
                self.weights.append(xavier_initialization(fan_in, fan_out))
            else:
                self.weights.append(np.random.randn(fan_in, fan_out) * 0.01)

            self.biases.append(np.zeros((1, fan_out)))

        self.activation_functions = []
        self.activation_derivatives = []
        for func in activation_functions:
            if func == "relu":
                self.activation_functions.append(relu)
                self.activation_derivatives.append(relu_derivative)
            elif func == "sigmoid":
                self.activation_functions.append(sigmoid)
                self.activation_derivatives.append(sigmoid_derivative)
            else:
                raise ValueError(f"Unsupported activation function: {func}")

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        # forward pass through hidden layers
        for i in range(self.num_layers - 1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            activation = self.activation_functions[i](z)
            self.activations.append(activation)

        # output layer (using softmax)
        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        output = softmax(z)
        self.activations.append(output)
        return output

    def backprop(self, x, y):
        m = y.shape[0]
        delta = self.activations[-1] - y
        d_weights = [None] * self.num_layers
        d_biases = [None] * self.num_layers

        # gradients for the output layer
        d_weights[-1] = self.activations[-2].T @ delta / m
        d_biases[-1] = np.sum(delta, axis=0, keepdims=True) / m

        # backpropagate through hidden layers
        for i in reversed(range(self.num_layers - 1)):
            delta = (delta @ self.weights[i + 1].T) * self.activation_derivatives[i](self.z_values[i])
            d_weights[i] = self.activations[i].T @ delta / m
            d_biases[i] = np.sum(delta, axis=0, keepdims=True) / m

        # update weights and biases using the current learning rate
        for i in range(self.num_layers):
            self.weights[i] -= self.learning_rate * d_weights[i]
            self.biases[i] -= self.learning_rate * d_biases[i]

    def train(self, x, y, epochs, batch_size):
        best_accuracy = 0
        epochs_since_improvement = 0

        for epoch in range(epochs):
            permutation = np.random.permutation(x.shape[0])
            x_shuffled = x[permutation]
            y_shuffled = y[permutation]

            for i in range(0, x.shape[0], batch_size):
                X_batch = x_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                self.forward(X_batch)
                self.backprop(X_batch, y_batch)

            # compute training accuracy
            predictions = np.argmax(self.forward(x), axis=1)
            targets = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == targets)
            print(f'Epoch {epoch + 1}/{epochs} - Accuracy: {accuracy * 100:.2f}%')

            # check if there is an improvement in accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # decay the learning rate if no improvement is seen for `patience` epochs
            if epochs_since_improvement >= self.patience:
                self.learning_rate *= self.decay_factor
                print(f"Learning rate reduced to {self.learning_rate:.5f}")
                epochs_since_improvement = 0
