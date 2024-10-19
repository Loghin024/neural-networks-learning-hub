from utils.data_loader import download_mnist

#load data
train_data, train_label = download_mnist(is_train=True)
validate_data, validate_label = download_mnist(is_train=False)