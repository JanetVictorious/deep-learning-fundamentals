import numpy as np


def sigmoid(z):
    """Sigmoid activation function."""
    a = 1.0 / (1.0 + np.exp(-z))
    assert a.shape == z.shape
    activation_cache = z
    return a, activation_cache


def tanh(z):
    """Tanh activation function."""
    a = np.tanh(z)
    assert a.shape == z.shape
    activation_cache = z
    return a, activation_cache


def relu(z):
    """ReLU activation function."""
    a = np.maximum(0.0, z)
    assert a.shape == z.shape
    activation_cache = z
    return a, activation_cache


def leaky_relu(z):
    """Leaky-ReLU activation function."""
    a = np.maximum(0.01 * z, z)
    assert a.shape == z.shape
    activation_cache = z
    return a, activation_cache


def sigmoid_backward(da, activation_cache):
    """Backward propagation of sigmoid unit."""
    z = activation_cache
    s = 1.0 / (1.0 + np.exp(-z))
    dz = da * s * (1 - s)
    assert dz.shape == z.shape
    return dz


def tanh_backward(da, activation_cache):
    """Backward propagation of tanh unit."""
    z = activation_cache
    s = np.tanh(z)
    dz = da * (1.0 - np.power(s, 2))
    assert dz.shape == z.shape
    return dz


def relu_backward(da, activation_cache):
    """Backward propagation of ReLU unit."""
    z = activation_cache
    dz = np.array(da, copy=True)
    dz[z < 0.0] = 0.0
    assert dz.shape == z.shape
    return dz


def leaky_relu_backward(da, activation_cache):
    """Backward propagation of Leaky-ReLU unit."""
    z = activation_cache
    dz = np.array(da, copy=True)
    dz[z <= 0.0] = 0.01
    assert dz.shape == z.shape
    return dz
