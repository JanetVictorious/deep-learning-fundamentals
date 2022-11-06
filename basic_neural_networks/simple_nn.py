import numpy as np


def _sigmoid(z):
    """Sigmoid activation function."""
    a = 1 / (1 + np.exp(-z))
    return a


class OneLayerNN:
    def __init__(self):
        self.params = dict()
        self.learning_curve = []

    def _layer_sizes(self, X, y, hidden_units: int = 4):
        """Define layer sizes of network.

        :param X:
            Input layer, `(n_x, m)` array with `n_x` as nr of features
            and `m` as nr of examples.
        :param y:
            Output layer, `(1, m)` array with binary outcomes.
        :returns:
            Layer sizes of input, hidden, and output layer.
        """
        # Set size for input, hidden, and output layer
        n_x = X.shape[0]
        n_h = hidden_units
        n_y = y.shape[0]

        return n_x, n_h, n_y

    def _init_params(self, n_x, n_h, n_y):
        """Initialize parameters.

        :param n_x:
            Size of input layer, this equals to nr of features.
        :param n_h:
            Size of hidden layer.
        :param n_y:
            Size of output layer.
        :returns:
            Dictionary with initialized weights and biases for layers.
        """
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
        """Forward propagation in network.

        Hidden layer uses `tanh` activation, output layers uses `sigmoid` activation.

        :param X:
            Input matrix on dimension `(n_x, m)`.
        :param params:
            Dictionary with weights and bias for layers.
        :returns:
            Activation of last layer and dictionary with activations and linear functions.
        """
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

        return A2, cache

    def _cost(self, A2, y):
        """Compute cost of network.

        :param A2:
            Activation of output layer.
        :param y:
            Output array on dimension `(1, m)`.
        :returns:
            Cost of activation.
        """
        # Nr of examples
        m = float(y.shape[1])

        # Cross-entropy cost
        logprobs = -(y * np.log(A2) + (1 - y) * np.log(1 - A2))
        cost = np.sum(logprobs) / m

        # Squeeze to right dimension
        cost = float(np.squeeze(cost))

        return cost

    def _backward_prop(self, params, cache, X, y):
        """Backward propagation in network.

        Derivatives are based on `tanh` and `sigmoid` activation functions.

        :param params:
            Dictionary with weights and biases of layers.
        :param cache:
            Dictionary with linear combinations and activations.
        :param X:
            Input features array.
        :param y:
            Output array.
        :returns:
            Dictionary with gradients of weights and biases.
        """
        # Nr of examples
        m = float(X.shape[1])

        # Retrieve weigths
        W2 = params['W2']

        # Retrieve forward propagation outputs
        A1 = cache['A1']
        A2 = cache['A2']

        # Backward propagation
        dZ2 = A2 - y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Gradients
        grads = {'dW1': dW1,
                 'db1': db1,
                 'dW2': dW2,
                 'db2': db2}

        return grads

    def _upgrade_params(self, params, grads, learning_rate: float = 0.1):
        """Update parameters using gradient descent.

        :param params:
            Dictionary with weights and biases.
        :param grads:
            Dictionary with gradients of weights and biases.
        :param learning_rate:
            Learning rate used in gradient descent.
        :returns:
            Dictionary with updated weights and biases.
        """
        # Retrieve weights and biases
        W1 = np.array([i for i in params['W1']])
        b1 = params['b1']
        W2 = np.array([i for i in params['W2']])
        b2 = params['b2']

        # Retrieve gradients
        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']

        # Gradient descent
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

        params = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2}

        return params

    def call(self, X, y, n_h: int = 4, num_iter: int = 10000, learning_rate: float = 0.1, print_cost: bool = False):
        """Train network with 1 hidden layer.

        :param X:
            Input features array.
        :param y:
            Output array.
        :param n_h:
            Units in hidden layer.
        :param num_iter:
            Nr of iterations of gradient descent.
        :param learning_rate:
            Learning rate used in gradient descent.
        :param print_cost:
            If true, print cost.
        :returns:
            Self.
        """
        np.random.seed(42)

        # Input/Output layers
        n_x, n_h, n_y = self._layer_sizes(X, y, hidden_units=n_h)

        # Initialize parameters
        params = self._init_params(n_x, n_h, n_y)
        costs = []

        for i in range(num_iter):
            # Forward propagation
            A2, cache = self._forward_prop(X, params)

            # Cost
            cost = self._cost(A2, y)
            if i % 100 == 0:
                costs.append([i, cost])
            if print_cost and i % 1000 == 0:
                print(f'Cost after iteration {i}: {cost}')

            # Backward propagation
            grads = self._backward_prop(params, cache, X, y)

            # Gradient descent
            params = self._upgrade_params(params, grads, learning_rate)

        # Save learned parameters
        self.params = params

        # Save learning curve
        self.learning_curve = np.array(costs).reshape(-1, 2)

        return self

    def _predict(self, X):
        """Use learned parameters for prediction.

        :param X:
            Input features array.
        :returns:
            Array with probabilities from last activation.
        """
        # Verify learned parameters exist
        if self.params == dict():
            err_msg = 'No learned parameters. Train model using `.call()` method.'
            raise ValueError(err_msg)

        # Forward prop with learned parameters
        A2, _ = self._forward_prop(X, self.params)

        return A2

    def predict(self, X):
        """Binary prediction.

        :param X:
            Input features array.
        :returns:
            Binary array from last activation.
        """
        y_pred = self._predict(X)
        y_pred = (y_pred > 0.5)
        y_pred = np.array(y_pred, dtype=int)
        return y_pred

    def predict_proba(self, X):
        """Predict probabilities."""
        return self._predict(X)


if __name__ == '__main__':
    """Debugging"""
    import sklearn.datasets as dt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss, f1_score, accuracy_score

    # Generate synthetic data
    X, y = dt.make_classification(n_samples=2000, n_features=4, n_informative=2, n_redundant=1, random_state=42)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transpose data
    X_train = X_train.T
    X_test = X_test.T
    y_train = y_train.reshape(1, -1)
    y_test = y_test.reshape(-1, 1)

    # Instantiate model
    model = OneLayerNN()

    # Fit model
    model.call(X_train, y_train, learning_rate=0.1, print_cost=True)

    # Predict on test data
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Evaluation
    print(f'Accuracy: {accuracy_score(y_test, y_pred.T)}')
    print(f'Logloss: {log_loss(y_test, y_pred_proba.T)}')
    print(f'F1 score: {f1_score(y_test, y_pred.T)}')
