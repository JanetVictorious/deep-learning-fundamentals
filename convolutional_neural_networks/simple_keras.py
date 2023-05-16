import tensorflow as tf
import tensorflow.keras.layers as tfl


class KerasSeqClassifier:
    def __init__(self, input_shape: tuple):
        """Initialization of sequential classifier.

        Args:
            input_shape (tuple): Input shape of data, e.g. (28, 28, 3).

        Returns:
            None.
        """
        self.input_shape = input_shape
        self.model = None

    def _make_classifier(self) -> tf.keras.Sequential:
        """Sequential model for multi-class classification model.

        Args:
            None.

        Returns:
            A compiled tf.keras.Sequential model.
        """
        model = tf.keras.Sequential([
            # ZeroPadding2D with padding 3
            tf.keras.Input(shape=self.input_shape),
            tfl.ZeroPadding2D(padding=3),

            # Conv2D with 32 7x7 filters and stride of 1
            tfl.Conv2D(filters=32, kernel_size=(7, 7), strides=1),

            # BatchNormalization for axis 3
            tfl.BatchNormalization(axis=3),

            # ReLU activation
            tfl.ReLU(),

            # Max Pooling 2D with default parameters
            tfl.MaxPool2D(),

            # Flatten layer
            tfl.Flatten(),

            # Dense layer with 1 unit for output and sigmoid activation
            tfl.Dense(10, activation='softmax')
        ])

        return model

    def _build_model(self) -> tf.keras.Sequential:
        """Build model.

        Args:
            None.

        Returns:
            A compiled tf.keras.Sequential model.
        """
        # Instantiate model
        model = self._make_classifier()

        # Compile model
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        # Model summary
        model.summary()

        return model

    def call(self, ds_train: tf.data.Dataset, ds_eval: tf.data.Dataset, epochs: int) -> None:
        """Train classifier.

        Args:
            ds_train (tf.data.Dataset): Train dataset.
            ds_eval (tf.data.Dataset): Validation dataset.
            epochs (int): Nr of epochs to train.

        Returns:
            None
        """

        # Build model
        model = self._build_model()

        # Train classifier
        model.fit(ds_train, epochs=epochs, validation_data=ds_eval)

        self.model = model

    def evaluate(self, ds_test) -> None:
        """Evaluate trained model.

        Args:
            ds_test (tf.data.Dataset): Test dataset.

        Returns:
        """
        self.model.evaluate(ds_test)
