import numpy as np


def _sigmoid(z):
    """Sigmoid activation function."""
    a = 1 / (1 + np.exp(-z))
    return a


class LogisticModel:
    def __init__(self):
        self.params = dict()
        self.learning_rate = []

    def _init_params(self, n_x):
        """Initialize weights and bias."""
        W = np.random.randn(n_x, 1) * 0.01  # Initialize weights randomly
        b = 0.0

        params = {'W': W, 'b': b}

        return params

    def _log_loss(self, y, y_pred):
        """Cost function."""
        assert y.shape == y_pred.shape
        cost = np.mean(-(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))

        return cost

    def _forward_prop(self, X, W, b):
        """Forward propagation."""
        # Linear function of weights, input, and bias
        Z = np.dot(X, W) + b

        # Forward pass with sigmoid activation function
        A = _sigmoid(Z)

        assert A.shape == (X.shape[0], 1)

        return A

    def _backward_prop(self, X, A, y):
        """Backward propagation."""
        # Nr examples
        m = float(X.shape[0])

        # Calculate gradients
        dW = np.dot(X.T, (A - y)) / m
        db = np.mean(A - y)

        grads = {'dW': dW, 'db': db}

        return grads

    def _optimize(self, X, y, W, b, num_iter: int = 10000, learning_rate: float = 0.1, print_cost: bool = False):
        """Optimize weights and bias using gradient descent."""

        W_hat = np.array([i for i in W]).reshape(-1, 1)
        b_hat = b
        costs = []

        for i in range(num_iter):
            # Forward propagation
            A = self._forward_prop(X, W_hat, b_hat)

            # Cost
            cost = self._log_loss(y, A)
            if i % 100 == 0:
                costs.append(cost)
                if print_cost and i % 100 == 0:
                    print(f'Cost after iteration {i}: {cost}')

            # Backward propagation
            grads = self._backward_prop(X, A, y)
            dW = grads['dW']
            db = grads['db']

            assert W_hat.shape == dW.shape

            # Gradient descent
            W_hat = W_hat - learning_rate * dW
            b_hat = b_hat - learning_rate * db

        # Updated params
        params = {'W': W_hat, 'b': b_hat}
        grads = {'dW': dW, 'db': db}

        return params, grads, costs

    def fit(self, X, y, **kwargs):
        """Train logistic regression model."""
        y = y.reshape(-1, 1)

        # Nr examples and features
        n_x = X.shape[1]
        m = X.shape[0]

        assert y.shape == (m, 1)

        # Initialize params
        params = self._init_params(n_x)
        W = params['W']
        b = params['b']

        assert W.shape[0] == n_x

        # Optimize params
        params, _, costs = self._optimize(X, y, W, b, **kwargs)

        # Set params to model
        self.params = params

        # Set learning rate to model
        self.learning_rate = costs

        return self

    def _predict(self, X):
        """Forward pass with learned params."""
        if self.params == dict():
            err_msg = 'No parameters have been trainen for model. Use `.fit()` to train.'
            raise ValueError(err_msg)
        W = self.params['W']
        b = self.params['b']

        assert X.shape[1] == W.shape[0]

        y_pred = self._forward_prop(X, W, b)
        return y_pred

    def predict(self, X):
        """Binary prediction."""
        y_pred = self._predict(X)
        y_pred = np.array([i > 0.5 for i in y_pred], dtype=int).reshape(-1, 1)

        return y_pred

    def predict_proba(self, X):
        """Predict probabilities."""
        return self._predict(X)
