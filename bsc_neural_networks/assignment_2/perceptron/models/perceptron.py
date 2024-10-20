import numpy as np


class Perceptron:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        """
        Initialize perceptron model with input size, output size, and learning rate
        :param input_size:
        :param output_size:
        :param learning_rate:
        """
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.bias = np.zeros((output_size,))
        self.weights = np.random.randn(input_size, output_size) * 0.01

    def forward_propagation(self, input_data):
        """
        Perform forward propagation
        :param input_data:
        :return:
        """
        z_i = np.dot(input_data, self.weights) + self.bias
        z_i_exp = np.exp(
            z_i - np.max(z_i, axis=1, keepdims=True))  # trick to avoid numerical instability explained in readme
        return z_i_exp / np.sum(z_i_exp, axis=1, keepdims=True)

    def compute_loss(self, predicted_labels, Y):
        """
        Compute the cross-entropy loss between the predicted and actual labels.
        :param predicted_labels: Predicted labels
        :param Y: Actual labels
        :return: Cross-entropy loss
        """
        m = Y.shape[0]  # Number of samples
        # Cross-entropy loss
        loss = -np.sum(Y * np.log(predicted_labels + 1e-8)) / m
        return loss

    def backward(self, training_data, target_data, predicted_labels):
        """
        Perform backward propagation and update weights based on cross-entropy gradients.
        """

        error = target_data - predicted_labels  # (Target - Prediction)

        # update using gradient rule
        delta_w = np.dot(training_data.T, error)
        delta_b = np.sum(error, axis=0)

        # Update weights and biases using the learning rate
        self.weights += self.learning_rate * delta_w
        self.bias += self.learning_rate * delta_b
