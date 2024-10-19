from utils.data_loader import download_mnist

#load test data
test_data, test_label = download_mnist(is_train=False)

# print((test_data[0], test_label[0]))
print(test_label[0])