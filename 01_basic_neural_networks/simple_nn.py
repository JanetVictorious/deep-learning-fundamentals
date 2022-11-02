import numpy as np


def _sigmoid(z):
    """Sigmoid activation function."""
    a = 1 / (1 + np.exp(-z))
    return a


class OneLayerNN:
    def __init__(self):
        self.params = dict()
        self.learning_rate = []

    def _layer_sizes(self, X, y, hidden_units: int = 4):
        """Define layer sizes of network."""
        # Set size for input, hidden, and output layer
        n_x = X.shape[0]
        n_h = hidden_units
        n_y = y.shape[0]

        return n_x, n_h, n_y

    def _init_params(self, n_x, n_h, n_y):
        """Initialize parameters of layers."""
        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))

        # Set params
        params = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2}

        return params

    def _forward_prop(self, X, params):
        """Forward propagation in network."""
        # Retrieve parameters
        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']

        # Forward propagation
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = _sigmoid(Z2)

        # Cache values
        cache = {'Z1': Z1,
                 'A1': A1,
                 'Z2': Z2,
                 'A2': A2}

        return cache
