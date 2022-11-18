import numpy as np

from networks import BaseNetwork

np.random.seed(42)


class TestBaseNetwork:
    def test_init_params(self):
        bn = BaseNetwork()
        layer_dims = [2, 3, 1]
        params = bn._init_params(layer_dims)

        assert params['W1'].shape == (3, 2)
        assert params['b1'].shape == (3, 1)
        assert params['W2'].shape == (1, 3)
        assert params['b2'].shape == (1, 1)

    def test_linear_forward(self):
        bn = BaseNetwork()

        A = np.random.randn(2, 3)
        W = np.random.randn(4, 2)
        b = np.random.randn(4, 1)

        Z, linear_cache = bn._linear_forward(A, W, b)

        assert Z.shape == (4, 3)
        assert linear_cache == (A, W, b)

    def test_activation_forward(self):
        bn = BaseNetwork()

        A_prev = np.random.randn(2, 3)
        W = np.random.randn(4, 2)
        b = np.random.randn(4, 1)

        A, cache = bn._activation_forward(A_prev, W, b, 'sigmoid')
        linear_cache, activation_cache = cache

        assert A.shape == (W.shape[0], A_prev.shape[1])
        assert linear_cache == (A_prev, W, b)
        assert activation_cache.shape == A.shape

    def test_forwad_prop(self):
        bn = BaseNetwork()

        layer_dims = [2, 3, 2, 1]
        params = bn._init_params(layer_dims)

        X = np.array([[1.0, 0.5, 0.1], [0.9, 0.67, -0.44]]).reshape(2, -1)
        m = X.shape[1]

        AL, caches = bn._forward_prop(X, params, 'relu')

        assert AL.shape == (1, m)
        assert len(caches) == len(layer_dims) // 2 + 1

    def test_cost(self):
        bn = BaseNetwork()

        y1 = np.array([1, 0, 0, 1]).reshape(1, -1)
        AL1 = np.array([0.9, 0.3, 0.2, 0.88]).reshape(1, -1)

        y2 = np.array([1]).reshape(1, -1)
        AL2 = np.array([0.9]).reshape(1, -1)

        assert round(bn._cost(AL1, y1), 5) == 0.20325
        assert round(bn._cost(AL2, y2), 5) == 0.10536

    def test_linear_backward(self):
        assert True

    def test_activation_backward(self):
        assert True

    def test_backward_prop(self):
        assert True

    def test_upgrade_params(self):
        assert True

    def test_call(self):
        assert True

    def test_predict(self):
        assert True

    def test_predict_proba(self):
        assert True


class TestDeepNetworkInit:
    def test_one(self):
        assert True


class TestDeepNetworkReg:
    def test_one(self):
        assert True
