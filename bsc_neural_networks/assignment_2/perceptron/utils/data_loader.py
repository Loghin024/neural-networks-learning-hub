import numpy as np
from torchvision.datasets import MNIST

def download_mnist(is_train: bool):
    """
    Download MNIST dataset
    :param is_train:
    :return: data, labels
    """
    dataset = MNIST(root='./data',
                transform=lambda x: np.array(x).flatten(),
                download=True,
                train=is_train)

    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)

    return mnist_data, mnist_labels

def encode_one_hot(labels, num_classes):
    """
    One-hot encode labels
    :param labels:
    :param num_classes:
    :return:  one-hot encoded labels
    """
    return np.eye(num_classes)[labels]

def normalize_pixel(pixel):
    """
    Normalize pixel values
    :param pixel:
    :return: normalized pixel
    """
    return pixel / 255.0


def normalize_image(data):
    """
    Normalize image data
    :param data:
    :return: normalized image data
    """
    return np.array([normalize_pixel(pixel) for pixel in data])
