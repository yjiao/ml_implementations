"""

various math functions

"""

import numpy as np


def sigmoid(x: np.array) -> np.array:
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


def mse(y, y_hat):
    return np.sum((y - y_hat) ** 2) / len(y)


def mse_grad(y, y_hat):
    return y_hat - y
