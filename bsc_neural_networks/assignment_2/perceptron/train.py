from utils.data_loader import download_mnist, encode_one_hot, normalize_image

#load data
train_data, train_label = download_mnist(is_train=True)
validate_data, validate_label = download_mnist(is_train=False)

#normalize data
train_data = normalize_image(train_data)
validate_data = normalize_image(validate_data)

print(train_label[0])
#one-hot encode the labels
train_label = encode_one_hot(train_label, 10)
validate_label = encode_one_hot(validate_label, 10)

print((train_data[0], train_label[0]))