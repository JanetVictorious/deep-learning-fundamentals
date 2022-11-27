# Basic neural networks

These files and notebooks contain examples of basic approaches to neural networks.

---

## Logistic regression

In [logistic_model.py](./logistic_model.py) is the logic for a standard logistic regression implemented. The methods are structured in a way to fit the neural network approach of:

* Forward propagation
* Sigmoid activation
* Cost calculation
* Backward propagation
* Update of weights and bias using gradient descent

See notebook [logistic_regression_nn_approach](./logistic_regression_nn_approach.ipynb) for example usage.

---

## Neural network with one hidden layer

In [simple_nn.py](./simple_nn.py) is the logic for a neural network with one hidden layer implemented. The methods are structured in a similar fashion as to [logistic_model.py](./logistic_model.py).

See notebook [one_hidden_nn](./one_hidden_nn.ipynb) for example usage. The model is compared to a standard logistic regression model to show how the neural network is better at separating non-linearities between different classes.

---

## Deep neural networks

In [networks.py](./networks.py) is the logic for deep neural networks implemented. Networks can be implemented with multiple layers and units by providing a variable `layer_dims`. Initialization of weights and regularization methods are available as well.

### Deep network

See notebook [cat_vs_nocat_dnn](./cat_vs_nocat_dnn.ipynb) for example usage of a fully connected deep neural network applied on cat images.

### Weight initialization

Three methods are implemented:

* zero initialization, setting all weights to zero
* random initialization, setting weights randomly from a normal distribution and scaling with 0.01
* He initialization, setting weights randomly and scaling with the square root of 2 divided by the nr of nodes in the previous layer

See notebook [dnn_w_initialization](./dnn_w_initialization.ipynb) for example usage.

### Regularization

Two types of regularization is available:

* L2 regularization, penalizing weights in the cost function
* drop-out, randomly skip connections in a hidden layer

See notebook [dnn_w_regularization](./dnn_w_regularization.ipynb) for example usage.

---

## Deep neural networks with optimization

In [networks_w_optimization.py](./networks_w_optimization.py) is a network implemented which can perform optimization as follows:

* gradient descent
* momentum
* Adam
* learning rate decay

See notebook [dnn_w_optimization](./dnn_w_optimization.ipynb) for example usage.
