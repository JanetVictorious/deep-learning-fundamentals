import numpy as np

from utils.dnn_utils import sigmoid, relu, sigmoid_backward, relu_backward


class DeepNNModel:
    def __init__(self):
        self.layer_dims = []
        self.params = dict()
        self.learning_curve = []

    def _init_params(self, layer_dims):
        """Initialize parameters for deep network."""
        np.random.seed(42)
        params = dict()

        # Layers of network
        L = len(layer_dims)

        for i in range(1, L):
            params['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
            params['b' + str(i)] = np.zeros((layer_dims[i], 1))

            assert params['W' + str(i)].shape == (layer_dims[i], layer_dims[i - 1])
            assert params['b' + str(i)].shape == (layer_dims[i], 1)

        self.layer_dims = layer_dims

        return params

    def _linear_forward(self, A, W, b):
        """Linear part of forward propagation."""
        Z = np.dot(W, A) + b
        cache = (A, W, b)

        return Z, cache

    def _activation_forward(self, A_prev, W, b, activation):
        """Linear activation forward."""
        # Linear forward
        Z, linear_cache = self._linear_forward(A_prev, W, b)

        # Activation
        if activation == 'sigmoid':
            A, activation_cache = sigmoid(Z)
        elif activation == 'relu':
            A, activation_cache = relu(Z)
        else:
            err_msg = f'Activation {activation} not implemented.'
            raise ValueError(err_msg)

        # Cache linear and activation
        # Contains: A_prev, W, b, and Z
        cache = (linear_cache, activation_cache)

        return A, cache

    def _forward_prop(self, X, params):
        """Forward propagation in network."""
        caches = []
        A = X
        L = len(params) // 2

        # Forward propagation in hidden layers
        for i in range(1, L):
            A_prev = A
            A, cache = self._activation_forward(A_prev,
                                                params['W' + str(i)],
                                                params['b' + str(i)],
                                                activation='relu')
            caches.append(cache)

        # Forward propagation in output layer
        AL, cache = self._activation_forward(A,
                                             params['W' + str(L)],
                                             params['b' + str(L)],
                                             activation='sigmoid')
        caches.append(cache)

        return AL, caches

    def _cost(self, A, y):
        """Cost of network."""
        # Nr of examples
        m = float(y.shape[1])

        # Cross-entropy cost
        logprobs = -(y * np.log(A) + (1 - y) * np.log(1 - A))
        cost = np.sum(logprobs) / m

        # Squeeze to right dimension
        cost = float(np.squeeze(cost))

        return cost

    def _linear_backward(self, dZ, cache):
        """Linear part of backward propagation."""
        # Cached properties
        A_prev, W, b = cache

        # Nr of examples
        m = float(A_prev.shape[1])

        # Linear backward
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def _activation_backward(self, da, cache, activation):
        """Linear activation backward."""
        # Linear cache (A_prev, W, b) and activation cache (Z)
        linear_cache, activation_cache = cache

        # Backward activation
        if activation == 'sigmoid':
            dz = sigmoid_backward(da, activation_cache)
        elif activation == 'relu':
            dz = relu_backward(da, activation_cache)
        else:
            err_msg = f'Activation {activation} is not implemented.'
            raise ValueError(err_msg)

        # Linear backward
        da_prev, dw, db = self._linear_backward(dz, linear_cache)

        return da_prev, dw, db

    def _backward_prop(self, AL, y, caches):
        """Backward propagation in network."""
        # Dictionary for storing gradients
        grads = dict()
        L = len(caches)
        y = y.reshape(AL.shape)

        dAL = -(np.divide(y, AL) - np.divide(1 - y, 1 - AL))

        current_cache = caches[L - 1]
        da_prev_temp, dw_temp, db_temp = self._activation_backward(dAL, current_cache, 'sigmoid')
        grads['dA' + str(L - 1)] = da_prev_temp
        grads['dW' + str(L)] = dw_temp
        grads['db' + str(L)] = db_temp

        for i in reversed(range(L - 1)):
            current_cache = caches[i]
            da_prev_temp, dw_temp, db_temp = self._activation_backward(da_prev_temp, current_cache, 'relu')
            grads['dA' + str(i)] = da_prev_temp
            grads['dW' + str(i + 1)] = dw_temp
            grads['db' + str(i + 1)] = db_temp

        return grads

    def _upgrade_params(self, params, grads, learning_rate: float = 0.1):
        """Upgrade parameters using gradient descent."""
        parameters = {i: v for i, v in params.items()}
        L = len(parameters) // 2

        for i in range(L):
            parameters['W' + str(i + 1)] = parameters['W' + str(i + 1)] - learning_rate * grads['dW' + str(i + 1)]
            parameters['b' + str(i + 1)] = parameters['b' + str(i + 1)] - learning_rate * grads['db' + str(i + 1)]

        return parameters

    def call(self, X, y, layer_dims, num_iter: int = 10000, learning_rate: float = 0.1, print_cost: bool = False):
        """Train deep neural network."""
        np.random.seed(42)

        params = self._init_params(layer_dims)
        costs = []

        for i in range(num_iter):
            AL, caches = self._forward_prop(X, params)

            cost = self._cost(AL, y)
            if i % 100 == 0:
                costs.append([i, cost])
            if print_cost and i % 1000 == 0:
                print(f'Cost after iteration {i}: {cost}')

            grads = self._backward_prop(AL, y, caches)

            params = self._upgrade_params(params, grads, learning_rate)

        self.params = params

        self.learning_curve = np.array(costs).reshape(-1, 2)

        return self

    def _predict(self, X):
        """Use learned parameters for prediction."""
        if self.params == dict():
            err_msg = 'No learned parameters. Train model using `.cal()` method.'
            raise ValueError(err_msg)

        AL, _ = self._forward_prop(X, self.params)

        return AL

    def predict(self, X):
        """Binary prediction."""
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
    X, y = dt.make_classification(n_samples=2000, n_features=200, n_informative=100, n_redundant=100, random_state=42)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transpose data
    X_train = X_train.T
    X_test = X_test.T
    y_train = y_train.reshape(1, -1)
    y_test = y_test.reshape(-1, 1)

    # Instantiate model
    model = DeepNNModel()
    layer_dims = [X_train.shape[0], 20, 7, 5, y_train.shape[0]]  # 4-layer model

    # Fit model
    model.call(X_train, y_train, layer_dims, learning_rate=0.01, print_cost=True)

    # Predict on test data
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Evaluation
    print(f'Accuracy: {accuracy_score(y_test, y_pred.T)}')
    print(f'Logloss: {log_loss(y_test, y_pred_proba.T)}')
    print(f'F1 score: {f1_score(y_test, y_pred.T)}')
