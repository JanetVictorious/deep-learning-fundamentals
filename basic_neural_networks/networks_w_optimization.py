import numpy as np

from networks import DeepNetworkReg


class DeepNetworkOptim(DeepNetworkReg):
    def __init__(self):
        super().__init__()

    def _mini_batches(self, X, y, mini_batch_size=64, seed=42):
        """Split data into mini-batches.

        :param X:
            Input data of shape `(nr_features, nr_examples)`.
        :param y:
            Output label of shape `(1, nr_examples)`.
        :param mini_batch_size:
            Size of mini-batches.
        :returns:
            List of `(mini_batch_X, mini_batch_y)`.
        """
        # Set seed
        np.random.seed(seed)

        # Nr of examples and mini-batches list
        m = X.shape[1]
        mini_batches = []

        # Shuffle data
        shuffled_idx = list(np.random.permutation(m))
        X_rand = X[:, shuffled_idx]
        y_rand = y[:, shuffled_idx].reshape(1, m)

        # Mini-batch size as increment
        inc = mini_batch_size

        # Nr of full increments over data
        num_complete_mini_batches = int(m / inc)

        # Iterate over full increments and store mini-batches
        for i in range(0, num_complete_mini_batches):
            mini_batch_X = X_rand[:, i*inc:(i+1)*inc]
            mini_batch_y = y_rand[:, i*inc:(i+1)*inc]
            mini_batch = (mini_batch_X, mini_batch_y)
            mini_batches.append(mini_batch)

        # If data doesn't fill up increment last increment
        if m % inc != 0:
            mini_batch_X = X_rand[:, (i+1)*inc:]
            mini_batch_y = y_rand[:, (i+1)*inc:]
            mini_batch = (mini_batch_X, mini_batch_y)
            mini_batches.append(mini_batch)

        return mini_batches

    def _init_velocity(self, params):
        """Initialize velocity.

        :param params:
            Dictionary containing weights and biases.
        :returns:
            Dictionary with initialized velocity for weights and biases.
        """
        # Nr layers in network
        L = len(params) // 2
        v = dict()

        # Iterate over layers
        for i in range(1, L+1):
            v['dW' + str(i)] = np.zeros(params['W' + str(i)].shape)
            v['db' + str(i)] = np.zeros(params['b' + str(i)].shape)

        return v

    def _upgrade_params_momentum(self, params, grads, v, beta, learning_rate):
        """Upgrade parameters using momentum.

        :param params:
            Dictionary with weights and biases for layers.
        :param grads:
            Dictionary with gradients of weights and biases
            for layers.
        :param v:
            Dictionary with velocity for layers.
        :param beta:
            Momentum hyperparameter.
        :param learning_rate:
            Learning rate hyperparameter.
        :returns:
            Updated parameters and velocity.
        """
        # Nr of layers to iterate over
        L = len(params) // 2

        # Iterate over layers
        for i in range(1, L+1):
            # Update velocity
            v['dW' + str(i)] = beta * v['dW' + str(i)] + (1 - beta) * grads['dW' + str(i)]
            v['db' + str(i)] = beta * v['db' + str(i)] + (1 - beta) * grads['db' + str(i)]

            # Update parameters
            params['W' + str(i)] = params['W' + str(i)] - learning_rate * v['dW' + str(i)]
            params['b' + str(i)] = params['b' + str(i)] - learning_rate * v['db' + str(i)]

        return params, v

    def _init_adam(self, params):
        """Initialize Adam.

        :param params:
            Dictionary containing weights and biases.
        :returns:
            Dictionary with initialized exponentially weighted average of gradients.
            Dictionary with initialized exponentially weighted average of the square of gradients.
        """
        # Nr layers in network
        L = len(params) // 2

        # Dictionaries for first and second momentum
        v = dict()
        s = dict()

        # Iterate over layers
        for i in range(1, L+1):
            # Initialize first momentun
            v['dW' + str(i)] = np.zeros(params['W' + str(i)].shape)
            v['db' + str(i)] = np.zeros(params['b' + str(i)].shape)

            # Initialize second momentum
            s['dW' + str(i)] = np.zeros(params['W' + str(i)].shape)
            s['db' + str(i)] = np.zeros(params['b' + str(i)].shape)

        return v, s

    def _upgrade_params_adam(self, params, grads, v, s, t,
                             beta1=0.9, beta2=0.999, learning_rate=0.01, epsilon=1e-8):
        """Upgrade parameters using Adam.

        :param params:
            Dictionary with weights and biases.
        :param grads:
            Dictionary with gradients.
        :param v:
            Dictionary with exponentially weighted averages of gradients.
        :param s:
            Dictionary with exponentially weighted averages of square gradients.
        :param t:
            Nr of steps taken.
        :param beta1:
            Exponential decay for first moment estimates.
        :param beta2:
            Exponential decay for second moment estimates.
        :param learning_rate:
            Learning rate.
        :param epsilon:
            Error term preventing division by zero.
        :returns:
            Updated parameters, first and second momentum, first and second corrected momentum.
        """
        # Nr layers in network
        L = len(params) // 2

        # Corrected first and second momentum
        v_corr = dict()
        s_corr = dict()

        # Iterate over layers
        for i in range(1, L+1):
            # Moving-average of gradients
            v['dW' + str(i)] = beta1 * v['dW' + str(i)] + (1 - beta1) * grads['dW' + str(i)]
            v['db' + str(i)] = beta1 * v['db' + str(i)] + (1 - beta1) * grads['db' + str(i)]

            # Bias-corrected first moment estimate
            v_corr['dW' + str(i)] = v['dW' + str(i)] / (1 - beta1**t)
            v_corr['db' + str(i)] = v['db' + str(i)] / (1 - beta1**t)

            # Moving-average of squared gradients
            s['dW' + str(i)] = beta2 * s['dW' + str(i)] + (1 - beta2) * np.square(grads['dW' + str(i)])
            s['db' + str(i)] = beta2 * s['db' + str(i)] + (1 - beta2) * np.square(grads['db' + str(i)])

            # Bias-corrected second raw moment estimate
            s_corr['dW' + str(i)] = s['dW' + str(i)] / (1 - beta2**t)
            s_corr['db' + str(i)] = s['db' + str(i)] / (1 - beta2**t)

            # Update parameters
            params['W' + str(i)] = params['W' + str(i)] - learning_rate * v_corr['dW' + str(i)] / (np.sqrt(s_corr['dW' + str(i)]) + epsilon)  # noqa: E501
            params['b' + str(i)] = params['b' + str(i)] - learning_rate * v_corr['db' + str(i)] / (np.sqrt(s_corr['db' + str(i)]) + epsilon)  # noqa: E501

        return params, v, s, v_corr, s_corr

    def call(self, X, y, layer_dims,
             initialization='he',
             optimizer='gd',
             hidden_activation='relu',
             mini_batch_size=64,
             beta=0.9,
             beta1=0.9,
             beta2=0.999,
             epsilon=1e-8,
             num_epochs=5000,
             learning_rate=0.01,
             lbd=0.0,
             keep_prob=1.0,
             print_cost=False):
        """Train network."""
        np.random.seed(42)

        self.hidden_activation = hidden_activation

        # Initialize parameters
        params = self._init_params(layer_dims, initialization)
        m = float(X.shape[1])
        costs = []
        t = 0
        seed = 1

        # Set optimization strategy
        if optimizer == 'gd':
            pass  # No initialization needed for moments
        elif optimizer == 'momentum':
            v = self._init_velocity(params)
        elif optimizer == 'adam':
            v, s = self._init_adam(params)

        for i in range(num_epochs):
            # Define the random minibatches
            # Increment seed to reshuffle differently after each epoch
            seed += 1
            cost_total = 0.0
            mini_batches = self._mini_batches(X, y, mini_batch_size, seed)

            for mini_batch in mini_batches:
                mini_X, mini_y = mini_batch

                # Forward propagation
                AL, caches = self._forward_prop(mini_X, params, hidden_activation, keep_prob)

                # Cost
                cost = self._cost_w_regulatization(AL, mini_y, params, lbd)
                cost_total += cost

                # Backward propagation
                grads = self._backward_prop(AL, y, caches, hidden_activation, lbd, keep_prob)

                # Update parameters
                if optimizer == 'gd':
                    params = self._upgrade_params(params, grads, learning_rate)
                elif optimizer == 'momentum':
                    params, v = self._upgrade_params_momentum(params, grads, v, beta)
                elif optimizer == 'adam':
                    t += 1  # Adam counter
                    params, v, s = self._upgrade_params_adam(params, grads, v, s, t,
                                                             beta1, beta2, learning_rate, epsilon)

            # Average total cost
            cost_avg = cost_total / m

            # Save/Print costs
            if i % 100 == 0 or i == num_epochs - 1:
                costs.append(cost_avg)
            if print_cost and (i % 1000 == 0 or i == num_epochs - 1):
                print(f'Cost after epoch {i}: {cost_avg}')

        # Save learned parameters
        self.params = params

        # Save learning curve
        self.learning_curve = np.array(costs).reshape(-1, 2)
