import numpy as np

from utils.dnn_utils import (
    leaky_relu, leaky_relu_backward,
    relu, relu_backward,
    sigmoid, sigmoid_backward,
    tanh, tanh_backward,
)


class DeepNNModelReg:
    def __init__(self):
        self.layer_dims = []
        self.hidden_activation = None
        self.params = dict()
        self.learning_curve = []

    def _init_params(self, layer_dims):
        """Initialize parameters."""
        params = dict()
        L = len(layer_dims)

        for i in range(1, L):

            # He initialization of weights
            params['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(2 / layer_dims[i - 1])
            params['b' + str(i)] = np.zeros((layer_dims[i], 1))

            assert params['W' + str(i)].shape == (layer_dims[i], layer_dims[i - 1])
            assert params['b' + str(i)].shape == (layer_dims[i], 1)

        return params

    def _linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        linear_cache = (A, W, b)
        return Z, linear_cache

    def _activation_forward(self, A_prev, W, b, activation, keep_prob):
        Z, linear_cache = self._linear_forward(A_prev, W, b)
        if activation == 'sigmoid':
            A, activation_cache = sigmoid(Z)
        elif activation == 'tanh':
            A, activation_cache = tanh(Z)
        elif activation == 'relu':
            A, activation_cache = relu(Z)
        elif activation == 'leaky_relu':
            A, activation_cache = leaky_relu(Z)
        else:
            err_msg = f'Activation {activation} is not implemented.'
            raise ValueError(err_msg)

        # Drop-out logic
        D = np.random.rand(A.shape[0], A.shape[1])
        D = (D < keep_prob).astype(int)
        A = A * D
        A = A / keep_prob

        cache = (linear_cache, activation_cache, D)
        return A, cache

    def _forward_prop(self, X, params, hidden_activation, keep_prob):
        """Forward propagation in network"""
        # Input layer as first activation
        A = X

        # Length of layers in network
        L = len(params) // 2

        # List for storing caches for each layer
        caches = []

        # Forward propagation over all hidden layers
        for i in range(1, L):
            A_prev = A
            A, cache = self._activation_forward(A_prev, params['W' + str(i)], params['b' + str(i)],
                                                hidden_activation, keep_prob)
            caches.append(cache)

        # Forward propagation of output layer
        AL, cache = self._activation_forward(A, params['W' + str(L)], params['b' + str(L)], 'sigmoid', 1.0)
        caches.append(cache)

        return AL, caches

    def _cost(self, AL, y):
        """Compute cost of network."""
        # Nr of examples
        m = float(y.shape[1])

        # Handle probabilities which are 0 or 1
        # This is to avoid getting -Inf, or Inf
        epsilon = 1e-15
        AL[AL == 1.0] = 1.0 - epsilon
        AL[AL == 0.0] = epsilon

        # Cross-entropy cost
        logprobs = -(y * np.log(AL) + (1.0 - y) * np.log(1.0 - AL))
        cost = np.sum(logprobs) / m

        # Squeeze to right dimension
        cost = float(np.squeeze(cost))

        return cost

    def _cost_w_regulatization(self, AL, y, params, lbd):
        """Cost function with regulatization."""
        # Nr of examples
        m = float(y.shape[1])

        # Nr of layers to iterate over
        L = len(params) // 2

        # Cross-entropy cost
        cross_entropy_cost = self._cost(AL, y)

        # L2 regularized cost
        l2_cost = 0.0
        for i in range(1, L + 1):
            l2_cost += np.sum(np.square(params['W' + str(i)]))
        l2_cost = l2_cost * lbd / (2 * m)

        # Combine cross-entropy and l2 cost
        cost = cross_entropy_cost + l2_cost

        return cost

    def _linear_backward(self, dZ, linear_cache, lbd):
        """Linear backward with regularization."""
        A_prev, W, b = linear_cache
        m = float(A_prev.shape[1])
        dW = np.dot(dZ, A_prev.T) / m + lbd * W / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def _activation_backward(self, dA, cache, activation, lbd, keep_prob):
        linear_cache, activation_cache, _ = cache
        if activation == 'sigmoid':
            dZ = sigmoid_backward(dA, activation_cache)
        elif activation == 'tanh':
            dZ = tanh_backward(dA, activation_cache)
        elif activation == 'relu':
            dZ = relu_backward(dA, activation_cache)
        elif activation == 'leaky_relu':
            dZ = leaky_relu_backward(dA, activation_cache)
        else:
            err_msg = f'Activation {activation} is not implemented.'
            raise ValueError(err_msg)

        # Drop-out
        dA_prev, dW, db = self._linear_backward(dZ, linear_cache, lbd)
        dA_prev = dA_prev * D_prev / keep_prob

        return dA_prev, dW, db

    def _backward_prop(self, AL, y, caches, hidden_activation, lbd, keep_prob):
        """Backward propagation in network with regularization."""
        grads = dict()
        L = len(caches)
        y = y.reshape(AL.shape)

        # Derivative of cost function w.r.t activation
        dAL = AL - y
        assert dAL.shape == AL.shape

        # Derivatives from output layer
        cache = caches[L - 1]
        dA_prev, dW, db = self._activation_backward(dAL, cache, 'sigmoid', lbd, keep_prob)
        grads['dA' + str(L - 1)] = dA_prev
        grads['dW' + str(L)] = dW
        grads['db' + str(L)] = db

        # Derivatives of hidden layers
        for i in reversed(range(1, L)):
            cache = caches[i - 1]
            dA_prev, dW, db = self._activation_backward(dA_prev, cache, hidden_activation, lbd, keep_prob)
            grads['dA' + str(i - 1)] = dA_prev
            grads['dW' + str(i)] = dW
            grads['db' + str(i)] = db

        return grads

    def _upgrade_params(self, params, grads, learning_rate):
        """Update parameters using gradient descent."""
        # Copy parameters with weights and biases
        parameters = {key: value for key, value in params.items()}

        # Nr of layers to iterate over
        L = len(parameters) // 2

        for i in range(1, L + 1):
            parameters['W' + str(i)] = parameters['W' + str(i)] - learning_rate * grads['dW' + str(i)]
            parameters['b' + str(i)] = parameters['b' + str(i)] - learning_rate * grads['db' + str(i)]

        return parameters

    def call(self, X, y,
             layer_dims,
             hidden_activation='relu',
             lbd=0.0,
             keep_prob=1.0,
             num_iter: int = 10000,
             learning_rate: float = 0.1,
             print_cost: bool = False):
        """Train network."""
        np.random.seed(42)

        self.hidden_activation = hidden_activation

        # Initialize parameters
        params = self._init_params(layer_dims)
        costs = []

        for i in range(num_iter):
            # Forward propagation
            AL, caches = self._forward_prop(X, params, hidden_activation, keep_prob)

            # Cost
            cost = self._cost_w_regulatization(AL, y, params, lbd)
            if i % 100 == 0 or i == num_iter - 1:
                costs.append([i, cost])
            if print_cost and (i % 10000 == 0 or i == num_iter - 1):
                print(f'Cost after iteration {i}: {cost}')

            # Backward propagation
            grads = self._backward_prop(AL, y, caches, hidden_activation, lbd, keep_prob)

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
        A2, _ = self._forward_prop(X, self.params, self.hidden_activation)

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
    from sklearn.metrics import accuracy_score, f1_score, log_loss
    from sklearn.model_selection import train_test_split

    # Generate synthetic data
    # X, y = dt.make_classification(n_samples=2000, n_features=20, n_informative=10, n_redundant=1, random_state=42)
    X, y = dt.make_classification(n_samples=2, n_features=2, n_informative=2, n_redundant=0, random_state=42)

    # # Train/Test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Transpose data
    # X_train = X_train.T
    # X_test = X_test.T
    # y_train = y_train.reshape(1, -1)
    # y_test = y_test.reshape(-1, 1)

    X_train = X.T
    y_train = y.reshape(1, -1)

    # Instantiate model
    model = DeepNNModelReg()
    layer_dims = [X_train.shape[0], 2, 2, y_train.shape[0]]

    # Fit model
    model.call(X_train, y_train, layer_dims, hidden_activation='relu', lbd=0.0, keep_prob=1.0,
               num_iter=2500, learning_rate=0.5, print_cost=True)

    # # Predict on test data
    # y_pred = model.predict(X_test)
    # y_pred_proba = model.predict_proba(X_test)

    # # Evaluation
    # print(f'Accuracy: {accuracy_score(y_test, y_pred.T)}')
    # print(f'Logloss: {log_loss(y_test, y_pred_proba.T)}')
    # print(f'F1 score: {f1_score(y_test, y_pred.T)}')
