import numpy as np

def accuracy(y_true, y_pred):
    """Calculates accuracy by comparing true labels with predicted labels."""
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
