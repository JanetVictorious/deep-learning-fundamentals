import tensorflow as tf
import pandas as pd


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
            tf.keras.Input(shape=self.input_shape),

            # ZeroPadding2D with padding 3
            tf.keras.layers.ZeroPadding2D(padding=3),

            # Conv2D with 32 7x7 filters and stride of 1
            tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 7), strides=1),

            # BatchNormalization for axis 3
            tf.keras.layers.BatchNormalization(axis=3),

            # ReLU activation
            tf.keras.layers.ReLU(),

            # Max Pooling 2D with default parameters
            tf.keras.layers.MaxPool2D(),

            # Flatten layer
            tf.keras.layers.Flatten(),

            # Dense layer with 1 unit for output and sigmoid activation
            tf.keras.layers.Dense(10, activation='softmax')
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


class KerasFunctional:
    def __init__(self, input_shape: tuple, n_classes):
        """Initialization of funcional Keras model.

        Args:
            input_shape (tuple): Input shape of image.
            n_classes (int): Nr of classes for output.

        Returns:
            None.
        """
        self._input_shape = input_shape
        self._n_classes = n_classes
        self.history = None

    def _model(self) -> tf.keras.Model:
        """Keras functional model.

        Args:
            None.

        Returns:
            A tf.keras.Model.
        """
        input_img = tf.keras.Input(shape=self._input_shape)

        # CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
        z1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(4, 4), strides=1, padding='SAME')(input_img)

        # ReLU activation
        a1 = tf.keras.layers.ReLU()(z1)

        # MAXPOOL: window 8x8, stride 8, padding 'SAME'
        p1 = tf.keras.layers.MaxPool2D(pool_size=(8, 8), strides=8, padding='SAME')(a1)

        # CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
        z2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), strides=1, padding='SAME')(p1)

        # ReLU activation
        a2 = tf.keras.layers.ReLU()(z2)

        # MAXPOOL: window 4x4, stride 4, padding 'SAME'
        p2 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=4, padding='SAME')(a2)

        # Flatten
        f = tf.keras.layers.Flatten()(p2)

        # Dense layer
        output = tf.keras.layers.Dense(self._n_classes, activation='softmax')(f)

        model = tf.keras.Model(inputs=input_img, outputs=output)

        return model

    def _build_model(self) -> tf.keras.Model:
        """Build and compile model.

        Args:
            None.

        Returns:
            A compiled tf.keras.Model.
        """
        # Create model
        model = self._model()

        # Compile
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.summary()

        return model

    def call(self, ds_train: tf.data.Dataset, ds_eval: tf.data.Dataset, epochs: int) -> None:
        """Train model.

        Args:
            ds_train (tf.data.Dataset): Training dataset.
            ds_eval (tf.data.Dataset): Validation dataset.
            epochs (int): Epochs to train.

        Returns:
            None.
        """
        # Build model
        model = self._build_model()

        # Train model
        history = model.fit(ds_train, validation_data=ds_eval, epochs=epochs)

        self.history = history

    def pd_history(self) -> tuple:
        """History of training as pandas dataframe.

        Args:
            None.

        Returns:
            A tuple with pd.DataFrames.
        """
        # History as dataframe
        df = pd.DataFrame(self.history.history)

        # Separate loss and accuracy
        df_loss = df.loc[:, ['loss', 'val_loss']]
        df_acc = df.loc[:, ['accuracy', 'val_accuracy']]

        # Rename loss columns
        df_loss = df_loss.rename(columns={'loss': 'train', 'val_loss': 'validation'})
        df_acc = df_acc.rename(columns={'loss': 'train', 'val_loss': 'validation'})

        return df_loss, df_acc
