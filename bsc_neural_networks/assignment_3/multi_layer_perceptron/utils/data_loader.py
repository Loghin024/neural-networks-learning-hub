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

    return np.array(mnist_data), np.array(mnist_labels)
    # return mnist_data, mnist_labels