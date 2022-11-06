# Basic neural networks

These files and notebooks contain examples of the most basic approach to neural networks.

---

## Logistic regression

In [logistic_model.py](./logistic_model.py) is the logic for a standard logistic regression implemented. The methods are structured in a way to fit the neural network approach of:

* Forward propagation
* Sigmoid activation
* Cost calculation
* Backward propagation
* Upgrade of weights and bias using gradient descent

See notebook [logistic_regression_nn_approach](./logistic_regression_nn_approach.ipynb) for example usage.

---

## Neural network with one hidden layer

In [simple_nn.py](./simple_nn.py) is the logic for a neural network with one hidden layer implemented. The methods are structured in a similar fashion as to [logistic_model.py](./logistic_model.py).

See notebook [one_hidden_nn](./one_hidden_nn.ipynb) for example usage. The model is compared to a standard logistic regression model to show how the neural network is better at separating non-linearities between different classes.

---

## Deep neural network

In [deep_nn.py](./deep_nn.py) is the logic for a deep neural network implemented. The network can be implemented with multiple layers and units by providing a variable `layer_dims`. The methods are structured in a similar fashion as to [simple_nn.py](./simple_nn.py).

See notebook ... for example usage.
