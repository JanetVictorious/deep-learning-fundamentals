import numpy as np

from simple_cnn import (
    zero_pad,
    create_mask_from_window,
    distribute_value,
    BasicConvNetwork
)

np.random.seed(1)


def test_zero_pad():
    x = np.random.randn(4, 3, 3, 2)
    x_pad = zero_pad(x, 3)
    assert x_pad.shape == (4, 9, 9, 2)


def test_create_mask_from_widow():
    x = np.random.randn(2, 3)
    mask = create_mask_from_window(x)

    assert mask.shape == x.shape
    assert np.sum(mask) == 1


def test_distribute_value():
    x = 2
    shape = (2, 2)
    a = distribute_value(x, shape)

    assert type(a) == np.ndarray
    assert a.shape == shape
    assert np.sum(a) == x


class TestBasicConvNetwork:
    def test_conv_forward(self):
        bcn = BasicConvNetwork()

        A_prev = np.random.randn(2, 5, 7, 4)
        W = np.random.randn(3, 3, 4, 8)
        b = np.random.randn(1, 1, 1, 8)
        hparams = {'pad': 1, 'stride': 2}
        Z, cache = bcn._conv_forward(A_prev, W, b, hparams)

        assert Z.shape[1] == int((5 - 3 + 2 * hparams['pad']) / hparams['stride']) + 1
        assert Z.shape == (2, 3, 4, 8)
        assert len(cache) == 4

    def test_pool_forward(self):
        bcn = BasicConvNetwork()

        A_prev = np.random.randn(2, 5, 5, 3)
        hparams_1 = {'stride': 1, 'f': 3}
        hparams_2 = {'stride': 2, 'f': 3}
        A_1, cache_1 = bcn._pool_forward(A_prev, hparams_1)
        A_2, _ = bcn._pool_forward(A_prev, hparams_2)

        assert A_1.shape[1] == int((5 - hparams_1['f']) / hparams_1['stride']) + 1
        assert A_1.shape == (2, 3, 3, 3)
        assert A_2.shape == (2, 2, 2, 3)
        assert len(cache_1) == 3

    def test_conv_backward(self):
        bcn = BasicConvNetwork()

        A_prev = np.random.randn(10, 4, 4, 3)
        W = np.random.randn(2, 2, 3, 8)
        b = np.random.randn(1, 1, 1, 8)
        hparams = {'pad': 2, 'stride': 2}
        Z, cache_conv = bcn._conv_forward(A_prev, W, b, hparams)
        dA, dW, db = bcn._conv_backward(Z, cache_conv)

        assert dA.shape == (10, 4, 4, 3)
        assert dW.shape == (2, 2, 3, 8)
        assert db.shape == (1, 1, 1, 8)

    def test_pool_backward(self):
        np.random.seed(1)  # Force-set to make test successful in suite
        bcn = BasicConvNetwork()

        A_prev = np.random.randn(5, 5, 3, 2)
        hparams = {'stride': 1, 'f': 2}
        A_1, cache_1 = bcn._pool_forward(A_prev, hparams)
        A_2, cache_2 = bcn._pool_forward(A_prev, hparams, mode='avg')
        dA = np.random.randn(5, 4, 2, 2)

        dA_prev_1 = bcn._pool_backward(dA, cache_1)
        dA_prev_2 = bcn._pool_backward(dA, cache_2)

        assert dA_prev_1.shape == A_prev.shape
        assert np.allclose(dA_prev_1[1, 1], [[0., 0.],
                                             [5.05844394, -1.68282702],
                                             [0., 0.]])

        assert dA_prev_2.shape == A_prev.shape
        assert np.allclose(dA_prev_2[1, 1], [[0.08485462,  0.2787552],
                                             [1.26461098, -0.25749373],
                                             [1.17975636, -0.53624893]])
