import numpy as np

def accuracy(y_pred, y_true):
    # TODO only support Samples x labels y values
    return np.average(y_true[np.arange(y_pred.shape[0]), y_pred.argmax(1)]) * 100
