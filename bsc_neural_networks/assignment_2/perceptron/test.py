from models.perceptron import Perceptron
from utils.data_loader import download_mnist, normalize_image, encode_one_hot
from utils.metrics import accuracy
import numpy as np

# load test data
test_X, test_Y = download_mnist(is_train=False)
test_X = normalize_image(test_X)

# load trained perceptron model
input_size = 784  # 28x28 flattened images
output_size = 10  # 10 output classes
perceptron = Perceptron(input_size=input_size, output_size=output_size)

# load trained weights and biases
perceptron.weights = np.load('./models/weights.npy')
perceptron.bias = np.load('./models/bias.npy')

# inference
predictions = perceptron.forward_propagation(test_X)

# one-hot encode test labels
test_Y = encode_one_hot(test_Y, 10)

# calculate accuracy
print(f"Test Accuracy: {accuracy(test_Y, predictions):.4f}")
