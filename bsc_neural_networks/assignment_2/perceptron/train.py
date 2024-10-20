from models.perceptron import Perceptron
from utils.data_loader import download_mnist, normalize_image, encode_one_hot
from utils.metrics import accuracy
import numpy as np

# Load and prepare data
train_X, train_Y = download_mnist(is_train=True)
test_X, test_Y = download_mnist(is_train=False)

# Normalize the data
train_X = normalize_image(train_X)
test_X = normalize_image(test_X)

# Convert labels to one-hot encoding
train_Y = encode_one_hot(train_Y, 10)
test_Y = encode_one_hot(test_Y, 10)

# Initialize the model
input_size = train_X.shape[1]  # 784 input features (28x28 images flattened)
output_size = 10  # 10 classes for classification (digits 0-9)
learning_rate = 0.01
epochs = 100
batch_size = 100

perceptron = Perceptron(input_size=input_size, output_size=output_size, learning_rate=learning_rate)
prediction = perceptron.forward_propagation(test_X)
acc = accuracy(test_Y, prediction)
print(f"Accuracy before training: {acc:.4f}")

# Training loop
for epoch in range(epochs):
    epoch_loss = 0.0

    for i in range(0, train_X.shape[0], batch_size):
        X_batch = train_X[i:i + batch_size]
        Y_batch = train_Y[i:i + batch_size]

        # inference
        predicted_labels = perceptron.forward_propagation(X_batch)

        # compute loss
        loss = perceptron.compute_loss(predicted_labels, Y_batch)
        epoch_loss += loss

        # backward propagation
        perceptron.backward(X_batch, Y_batch, predicted_labels)

    # average loss for the epoch
    avg_epoch_loss = epoch_loss / (train_X.shape[0] // batch_size)
    print(f"Epoch {epoch + 1}, Loss: {avg_epoch_loss:.4f}")

    # accuracy evaluation
    if (epoch + 1) % 10 == 0:
        predictions = perceptron.forward_propagation(test_X)
        acc = accuracy(test_Y, predictions)
        print(f"Epoch {epoch + 1}, Test Accuracy: {acc:.4f}")

# Save the trained model
perceptron.W = np.load('./models/weights.npy')
perceptron.b = np.load('./models/bias.npy')
