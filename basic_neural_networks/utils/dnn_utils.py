import numpy as np


def sigmoid(z):
    """Sigmoid activation function."""
    a = 1 / (1 + np.exp(-z))
    cache = z
    return a, cache


def relu(z):
    """ReLU activation function."""
    a = np.maximum(0, z)
    assert a.shape == z.shape
    cache = z
    return a, cache


def sigmoid_backward(da, cache):
    """Backward propagation of sigmoid unit."""
    z = cache
    s = 1 / (1 + np.exp(-z))
    dz = da * s * (1 - s)
    assert dz.shape == z.shape
    return dz


def relu_backward(da, cache):
    """Backward propagation of ReLU unit."""
    z = cache
    dz = np.array(da, copy=True)
    dz[dz <= 0] = 0
    assert dz.shape == z.shape
    return dz
