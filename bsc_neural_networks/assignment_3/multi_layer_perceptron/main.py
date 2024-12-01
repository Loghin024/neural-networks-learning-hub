from models.mlp import MLP
from utils.data_loader import download_mnist
import numpy as np
from config import *
from utils.utils import *


def preprocess_data():

    # download mnist data
    x_train, y_train = download_mnist(is_train=True)
    x_test, y_test = download_mnist(is_train=False)

    # normalize data
    x_train = np.array(x_train) / 255.0
    x_test = np.array(x_test) / 255.0

    # One-hot encode the labels
    y_train = one_hot_encode(y_train, OUTPUT_SIZE)
    y_test = one_hot_encode(y_test, OUTPUT_SIZE)

    return x_train, y_train, x_test, y_test


def save_weights_and_biases(model, filename="model_parameters.npz"):
    parameters = {}
    for i, (w, b) in enumerate(zip(model.weights, model.biases)):
        parameters[f"weights_{i}"] = w
        parameters[f"biases_{i}"] = b

    np.savez(filename, **parameters)
    print(f"Weights and biases saved to {filename}.")


def load_weights_and_biases(filename="model_parameters.npz"):
    data = np.load(filename, allow_pickle=True)

    weights = [data[f"weights_{i}"] for i in range(len(data) // 2)]
    biases = [data[f"biases_{i}"] for i in range(len(data) // 2)]

    return weights, biases


# Main function
def main():

    x_train, y_train, x_test, y_test = preprocess_data()

    layer_sizes = [INPUT_SIZE] + HIDDEN_LAYERS + [OUTPUT_SIZE]
    activation_functions = ["relu"]

    model = MLP(layer_sizes, activation_functions)
    model.train(x_train, y_train, EPOCHS, BATCH_SIZE)
    save_weights_and_biases(model)

    # test the model on validation data
    predictions = np.argmax(model.forward(x_test), axis=1)
    targets = np.argmax(y_test, axis=1)
    test_accuracy = np.mean(predictions == targets)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


if __name__ == "__main__":
    main()
