import numpy as np
class Perceptron:
    def __init__(self, input_size, output_size, learning_rate):
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
        z_i_exp = np.exp(z_i - np.max(z_i, axis=1, keepdims=True)) #trick to avoid numerical instability explained in readme
        return z_i_exp / np.sum(z_i_exp, axis=1, keepdims=True)

    # def backward_propagation(self, input_data, output_data, ):