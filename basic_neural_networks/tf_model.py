import tensorflow as tf
import numpy as np


def normalize_img(image, label):
    """Transform image and normalize."""
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1, ])
    return image, label


class SimpleTfModel:
    def __init__(self):
        self.params = dict()
        self.train_acc = []
        self.test_acc = []

    def _init_params(self, input_shape=12288, output_shape=6):
        """Initialize parameters for network."""
        initializer = tf.keras.initializers.GlorotNormal(seed=1)
        W1 = tf.Variable(initializer(shape=(25, input_shape)))
        b1 = tf.Variable(initializer(shape=(25, 1)))
        W2 = tf.Variable(initializer(shape=(12, 25)))
        b2 = tf.Variable(initializer(shape=(12, 1)))
        W3 = tf.Variable(initializer(shape=(output_shape, 12)))
        b3 = tf.Variable(initializer(shape=(output_shape, 1)))

        params = {'W1': W1, 'b1': b1,
                  'W2': W2, 'b2': b2,
                  'W3': W3, 'b3': b3}

        return params

    def _forward_prop(self, X, params):
        """Forward propagation."""
        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']
        W3 = params['W3']
        b3 = params['b3']

        Z1 = tf.math.add(tf.linalg.matmul(W1, X), b1)
        A1 = tf.keras.activations.relu(Z1)
        Z2 = tf.math.add(tf.linalg.matmul(W2, A1), b2)
        A2 = tf.keras.activations.relu(Z2)
        Z3 = tf.math.add(tf.linalg.matmul(W3, A2), b3)

        return Z3

    def _cost(self, logits, y):
        """Cross-entropy cost for logits."""
        logits = tf.transpose(logits)
        y = tf.transpose(y)
        cost = tf.keras.losses.categorical_crossentropy(y, logits, from_logits=True)
        cost = tf.reduce_sum(cost)
        return cost

    def call(self, X_train, X_test, y_train=None, y_test=None, num_classes=None,
             learning_rate=0.01, num_epochs=1500, minibatch_size=32, print_cost=False):
        """Train network."""
        costs = []
        train_acc = []
        test_acc = []

        # Nr of training examples
        if isinstance(X_train, tf.data.Dataset) and not y_train and not y_test:
            m = X_train.cardinality().numpy()
            if not num_classes:
                unique_labels = set()
                for _, label in X_train:
                    unique_labels.add(label.numpy())
                num_classes = len(unique_labels)
        elif isinstance(X_train, np.ndarray) and y_train and y_test:
            m = X_train.shape[1]
            num_classes = len(set([i for i in y_train]))
        else:
            err_msg = f'Data is of type {type(X_train)} which is not compatible.'
            raise ValueError(err_msg)

        # Init parameters
        params = self._init_params(input_shape=m, output_shape=num_classes)

        # Extract parameters
        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']
        W3 = params['W3']
        b3 = params['b3']

        # Set Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        # Track accuracy of multi-class problem
        train_accuracy = tf.keras.metrics.CategoricalAccuracy()
        test_accuracy = tf.keras.metrics.CategoricalAccuracy()

        # Datasets
        if isinstance(X_train, tf.data.Dataset) and not y_train and not y_test:
            ds_train = X_train
            ds_test = X_test
        else:
            ds_train = tf.data.Dataset.zip((X_train, y_train))
            ds_test = tf.data.Dataset.zip((X_test, y_test))

        # Create mini-batches
        mini_batches = ds_train.batch(minibatch_size).prefetch(tf.data.AUTOTUNE)
        mini_batches_test = ds_test.batch(minibatch_size).prefetch(tf.data.AUTOTUNE)

        # Train over epochs
        for epoch in range(num_epochs):
            epoch_cost = 0.0

            # Reset accuracy for each epoch
            train_accuracy.reset_states()

            for (mini_X, mini_y) in mini_batches:
                with tf.GradientTape() as tape:
                    # Forward propagation
                    Z3 = self._forward_prop(mini_X, params)

                    # Compute cost
                    mini_batch_cost = self._cost(Z3, mini_y)

                # Update accuracy with mini-batch
                train_accuracy.update_state(mini_y, Z3)

                # Calculate gradients
                trainable_vars = [W1, b1, W2, b2, W3, b3]
                grads = tape.gradient(mini_batch_cost, trainable_vars)

                # Adam optimization
                optimizer.apply_gradients(zip(grads, trainable_vars))

                # Add cost
                epoch_cost += mini_batch_cost

            # Average cost over training examples
            epoch_cost /= m

            if print_cost and epoch % 10 == 0:
                print(f'Cost after epoch {epoch}: {epoch_cost}')
                print(f'Train accuracy: {train_accuracy.result()}')

                # Evaluate test set
                for (mini_X_test, mini_y_test) in mini_batches_test:
                    Z3 = self._forward_prop(mini_X_test, params)
                    test_accuracy.update_state(mini_y_test, Z3)
                print(f'Test accuracy: {test_accuracy.result()}')

                costs.append([epoch, epoch_cost])
                train_acc.append(train_accuracy.result())
                test_acc.append(test_accuracy.result())

        # Save learned paramters
        self.params = params

        # Save results
        self.train_acc = train_acc
        self.test_acc = test_acc

        return self


if __name__ == '__main__':
    """Debugging."""
    import tensorflow_datasets as tfds

    # Load mnist dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # Print info
    print(ds_info.splits['train'].num_examples)
    print(ds_train.element_spec)
    print(ds_info.features['label'].num_classes)

    # Prep data
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    print(ds_train.element_spec)
    print(ds_test.element_spec)
    ds_train = ds_train.cache()
    ds_test = ds_test.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_test = ds_test.shuffle(ds_info.splits['test'].num_examples)
    num_classes = ds_info.features['label'].num_classes
    # ds_train = ds_train.batch(128)
    # ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Pass through network
    model = SimpleTfModel()
    model.call(X_train=ds_train, X_test=ds_test, num_classes=num_classes,
               learning_rate=0.01, num_epochs=1500, minibatch_size=32, print_cost=False)
