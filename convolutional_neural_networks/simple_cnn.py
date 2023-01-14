import numpy as np


def zero_pad(X, pad):
    """Pad images of X with zeros.

    Args:
        X: np.array of shape (m, n_H, n_W, n_C).
        pad: Integer, amount of horizontal and vertical padding around each image.

    Returns:
        Padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C).
    """
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=(0, 0))
    return X_pad


def conv_single_step(prev_slice, W, b):
    """Apply one filter defined by paramters W on a single slice.

    Args:
        prev_slice: Slice of input data of shape (f, f, n_C_prev).
        W: Weight parameters contained in a window - matrix of shape (f, f, n_C_prev).
        b: Bias parameters contained in a window - matrix of shape (1, 1, 1).

    Returns:
        Scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data.
    """
    s = prev_slice * W
    Z = np.sum(s)
    Z = Z + float(b)
    return Z


def create_mask_from_window(x):
    """Creates a mask from an input matrix x, to identify the max of x.

    Args:
        x: Array of shape (f, f).

    Returns:
        Array of shape (f, f).
    """
    mask = (x == np.max(x))
    return mask


def distribute_value(dz, shape):
    """Distributes the input value in the matrix of dimension shape.

    Args:
        dz: Input scalar.
        shape: Shape (n_H, n_W) of output matrix.

    Returns:
        Array of size (n_H, n_W) for which we distributed the value of dz.
    """
    # Dimensions from shape
    (n_H, n_W) = shape

    # Compute value to distribute
    avg = float(dz / (n_H * n_W))

    # Create matrix avg entries
    a = np.ones(shape) * avg

    return a


class BasicConvNetwork:
    def __init__(self):
        self.params = None
        self.learning_curve = []

    def _conv_forward(self, A_prev, W, b, hparams):
        """Implements forward propagation for a convolutional function.

        Args:
            A_prev: Output activations of the previous layer - shape (m, n_H_prev, n_W_prev, n_C_prev).
            W: Weights - shape (f, f, n_C_prev, n_C).
            b: Bias - shape (1, 1, 1, n_C).
            hparams: Dictionary containing stride and padding.

        Returns:
            Convolution output - shape (m, n_H, n_W, n_C).
            cache - values needed for backward propagation.
        """
        # Dimensions of A_prev
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Dimensions of filter
        (f, f, n_C_prev, n_C) = W.shape

        # Hparams
        stride = hparams['stride']
        pad = hparams['pad']

        # Dimensions of output volume
        n_H = int(((n_H_prev + 2 * pad - f) / stride) + 1)
        n_W = int(((n_W_prev + 2 * pad - f) / stride) + 1)

        # Initialize the output volume Z with zeros
        Z = np.zeros((m, n_H, n_W, n_C))

        # Apply padding to input activations
        A_prev_pad = zero_pad(A_prev, pad)

        # Loop over training examples
        for i in range(m):
            # Select ith training example's padded activation
            a_prev_pad = A_prev_pad[i]

            # Loop over vertical axis of the output volume
            for h in range(n_H):
                # Vertical start and end of the current slice
                vert_start = stride * h
                vert_end = vert_start + f

                # Loop over horizontal axis of the output volume
                for w in range(n_W):
                    # Horizontal start and end of the current slice
                    horiz_start = stride * w
                    horiz_end = horiz_start + f

                    # Loop over channels (= #filters) of the output volume
                    for c in range(n_C):
                        # Use corners to define the (3D) slice of a_prev_pad
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron
                        weights = a_slice_prev * W[:, :, :, c]
                        biases = b[:, :, :, c]
                        Z[i, h, w, c] = np.sum(weights) + float(biases)

        # Save information for backward propagation
        cache = (A_prev, W, b, hparams)

        return Z, cache

    def _pool_forward(self, A_prev, hparams, mode: str = 'max'):
        """Forward pass of the pooling layer.

        Args:
            A_prev

        Returns:
            Something
        """
        # Dimensions of A_prev
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Hparams
        stride = hparams['stride']
        f = hparams['f']

        # Dimensions of output volume
        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev

        # Initialize output matrix A
        A = np.zeros((m, n_H, n_W, n_C))

        # Loop over training examples
        for i in range(m):
            # Loop over vertical axis of the output volume
            for h in range(n_H):
                # Vertical start and end of the current slice
                vert_start = stride * h
                vert_end = vert_start + f

                # Loop over horizontal axis of the output volume
                for w in range(n_W):
                    # Horizontal start and end of the current slice
                    horiz_start = stride * w
                    horiz_end = horiz_start + f

                    # Loop over channels of the output volume
                    for c in range(n_C):
                        # Use corners to define the (3D) slice of a_prev_pad
                        a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                        # Type of pooling
                        if mode == 'max':
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif mode == 'avg':
                            A[i, h, w, c] = np.mean(a_prev_slice)

        # Store the input and hparameters in "cache" for pool_backward()
        cache = (A_prev, hparams, mode)

        # Making sure output shape is correct
        assert A.shape == (m, n_H, n_W, n_C)

        return A, cache

    def _conv_backward(self, dZ, cache):
        """Backward pass of the convolutional layer.

        Arguments:
            dZ: Gradient of the cost with respect to the output of the conv layer (Z),
                numpy array of shape (m, n_H, n_W, n_C).
            cache: Cache of values needed for the backward pass, output of forward pass.

        Returns:
            dA_prev: Gradient of the cost with respect to the input of the conv layer (A_prev),
                numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev).
            dW: Gradient of the cost with respect to the weights of the conv layer (W)
                numpy array of shape (f, f, n_C_prev, n_C).
            db: Gradient of the cost with respect to the biases of the conv layer (b)
                numpy array of shape (1, 1, 1, n_C).
        """
        # Information from cache
        (A_prev, W, b, hparams) = cache

        # Dimensions of A_prev
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Dimensions of W
        (f, f, n_C_prev, n_C) = W.shape

        # hparams
        pad = hparams['pad']
        stride = hparams['stride']

        # Dimensions of dZ
        (m, n_H, n_W, n_C) = dZ.shape

        # Initialize dA_prev, dW, db
        dA_prev = np.zeros(A_prev.shape)
        dW = np.zeros(W.shape)
        db = np.zeros(b.shape)

        # Pad A_prev and dA_prev
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)

        # Loop over training examples
        for i in range(m):
            # ith training example from A_prev_pad and dA_prev_pad
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]

            # Loop over vertical axis of the output volume
            for h in range(n_H):
                # Loop over horizontal axis of the output volume
                for w in range(n_W):
                    # Loop over channels of the output volume
                    for c in range(n_C):
                        # Corners of current slice
                        vert_start = stride * h
                        vert_end = vert_start + f
                        horiz_start = stride * w
                        horiz_end = horiz_start + f

                        # Slice from a_prev_pad
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Update gradients for the window and the filter's parameters
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                        dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                        db[:, :, :, c] += dZ[i, h, w, c]

            # ith training example dA_prev to the unpadded da_prev_pad
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

        # Making sure output shape is correct
        assert dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev)

        return dA_prev, dW, db

    def _pool_backward(self, dA, cache):
        """Backward pass of the pooling layer.

        Arguments:
            dA: Gradient of cost with respect to the output of the pooling layer, same shape as A.
            cache: Cache output from the forward pass of the pooling layer, contains the layer's input,
                hparams, and mode.

        Returns:
            Gradient of cost with respect to the input of the pooling layer, same shape as A_prev.
        """
        # Information from cache
        (A_prev, hparams, mode) = cache

        # hparams
        stride = hparams['stride']
        f = hparams['f']

        # Dimensions of A_prev and dA
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dA.shape

        # Initialize dA_prev
        dA_prev = np.zeros(A_prev.shape)

        # Loop over training examples
        for i in range(m):
            # Training example from A_prev
            a_prev = A_prev[i]

            # Loop over vertical axis
            for h in range(n_H):
                # Loop over horizontal axis
                for w in range(n_W):
                    # Loop over channels
                    for c in range(n_C):
                        # Corners of current slice
                        vert_start = stride * h
                        vert_end = vert_start + f
                        horiz_start = stride * w
                        horiz_end = horiz_start + f

                        # Compute the backward propagation for mode
                        if mode == 'max':
                            # Slice from a_prev
                            a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]

                            # Create the mask from a_prev_slice
                            mask = create_mask_from_window(a_prev_slice)

                            # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]
                        elif mode == 'avg':
                            # Get the value da from dA
                            da = dA[i, h, w, c]

                            # Define the shape of the filter as fxf
                            shape = (f, f)

                            # Distribute to get the correct slice of dA_prev
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)

        # Making sure output shape is correct
        assert dA_prev.shape == A_prev.shape

        return dA_prev
