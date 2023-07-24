from typing import Tuple, Callable

import tensorflow as tf
# from tensorflow.keras.applications.resnet_v2 import ResNet50V2
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import layers
from tensorflow.keras.initializers import glorot_uniform

import pandas as pd


def resnet(input_shape: Tuple[int],
           stack_fn: Callable,
           classes: int,
           initializer: tf.keras.initializers = glorot_uniform,
           model_name: str = 'resnet'):
    """Implementation of ResNet.

    Args:
        input_shape (Tuple[int]): Input shape of images.
        stack_fn (Callable): A function that returns output tensor for the stacked residual blocks.
        classes (int): Number of classes to classify image.
        initializer (tf.keras.initializers): To set up the initial weights of a layer.
        model_name (str): Model name.

    Returns:
        A keras Model instance.
    """
    # Define the input as a tensor with shape input_shape
    img_input = layers.Input(input_shape)

    # Zero padding
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)

    # Stage 1
    x = layers.Conv2D(filters=64,
                      kernel_size=7,
                      strides=2,
                      kernel_initializer=initializer(seed=0),
                      name='conv1_conv')(x)
    x = layers.BatchNormalization(axis=3, name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, name='pool1_pool')(x)

    # Stacked blocks
    x = stack_fn(x)

    # Average pooling
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    # Output layer
    x = layers.Dense(units=classes,
                     activation='softmax',
                     kernel_initializer=initializer(seed=0),
                     name='predictions')(x)

    # Create model
    model = tf.keras.Model(img_input, x, name=model_name)

    return model


def block(x: tf.Tensor,
          f: int,
          filters: int,
          stride: int = 1,
          conv_shortcut: bool = True,
          training: bool = True,
          initializer: tf.keras.initializers = glorot_uniform,
          name: str = ''):
    """Implementation of the convolutional and identity block.

    Args:
        x (tf.Tensor): Input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev).
        f (int): Specifying the shape of the middle CONV's window for the main path.
        filters (int): Filters of the bottleneck layer.
        stride (int): Specifying the stride to be used in the first layer.
        conv_shortcut (bool): Use convolution shortcut if True, otherwise identity shortcut.
        training (bool): If true, behave in training mode, else behave in inference mode.
        initializer (tf.keras.initializers): To set up the initial weights of a layer.
        name (str): Block label.

    Returns:
        Output of the convolutional block, tensor of shape (m, n_H, n_W, n_C).
    """
    # Shortcut path
    if conv_shortcut:
        shortcut = layers.Conv2D(filters=4 * filters,
                                 kernel_size=1,
                                 strides=stride,
                                 padding='valid',
                                 kernel_initializer=initializer(seed=0),
                                 name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=3, name=name + '_0_bn')(shortcut, training=training)
    else:
        shortcut = x

    # First component of main path
    x = layers.Conv2D(filters=filters,
                      kernel_size=1,
                      strides=stride,
                      padding='valid',
                      kernel_initializer=initializer(seed=0),
                      name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=3, name=name + '_1_bn')(x, training=training)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    # Second component of main path
    x = layers.Conv2D(filters=filters,
                      kernel_size=f,
                      strides=1,
                      padding='same',
                      kernel_initializer=initializer(seed=0),
                      name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=3, name=name + '_2_bn')(x, training=training)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    # Third component of main path
    x = layers.Conv2D(filters=4 * filters,
                      kernel_size=1,
                      strides=1,
                      padding='valid',
                      kernel_initializer=initializer(seed=0),
                      name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=3, name=name + '_3_bn')(x, training=training)

    # Add shortcut value to main path, and pass it through a RELU activation
    x = layers.Add(name=name + '_add')([x, shortcut])
    x = layers.Activation('relu', name=name + '_out')(x)

    return x


def stack(x: tf.Tensor, filters: int, blocks: int, stride: int = 2, name: str = ''):
    """A set of stacked residual blocks.

    Args:
        x (tf.Tensor): Input tensor.
        filters (int): Filters of the bottle neck layer in a block.
        blocks (int): Blocks in the stacked blocks.
        stride (int): Stride of the first layer in the first block.
        name (str): Stack label.

    Returns:
        Output tensor for the stacked blocks.
    """
    x = block(x=x, f=3, filters=filters, stride=stride, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block(x=x,
                  f=3,
                  filters=filters,
                  stride=1,
                  conv_shortcut=False,
                  name=name + '_block' + str(i))
    return x


def resnet_50(input_shape: Tuple[int], classes: int):
    """ResNet50 implementation."""
    def stack_fn(x):
        x = stack(x=x, filters=64, blocks=3, stride=1, name='conv2')
        x = stack(x=x, filters=128, blocks=4, name='conv3')
        x = stack(x=x, filters=256, blocks=6, name='conv4')
        return stack(x=x, filters=512, blocks=3, name='conv5')

    return resnet(input_shape=input_shape,
                  stack_fn=stack_fn,
                  classes=classes)


class ResNet50:
    def __init__(self, input_shape: Tuple[int], classes: int):
        """Initialization of ResNet50.

        Args:
            input_shape (Tuple[int]): Input shape of image.
            classes (int): Number of classes to classify image.

        Returns:
            None.
        """
        self.input_shape = input_shape
        self.classes = classes

    def _model(self) -> tf.keras.Model:
        """ResNet50 model.

        Args:
            None.

        Returns:
            A tf.keras.Model of ResNet50.
        """

        return resnet_50(input_shape=self.input_shape, classes=self.classes)

    def _build_model(self, print_summary: bool = True) -> tf.keras.Model:
        """Build and compile model.

        Args:
            print_summary (bool): If summary of model should be printed.

        Returns:
            A compiled tf.keras.Model.
        """
        # Create model
        model = self._model()

        # Compile
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        if print_summary:
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

        self.model = model
        self.history = history

    def pd_history(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """History of training as pandas dataframe.

        Args:
            None.

        Returns:
            A tuple with pd.DataFrames.
        """
        df = pd.DataFrame(self.history.history)

        df_loss = df.loc[:, ['loss', 'val_loss']]
        df_acc = df.loc[:, ['accuracy', 'val_accuracy']]

        df_loss = df_loss.rename(columns={'loss': 'train', 'val_loss': 'validation'})
        df_acc = df_acc.rename(columns={'accuracy': 'train', 'val_accuracy': 'validation'})

        return df_loss, df_acc

    def evaluate(self, ds_test: tf.data.Dataset):
        """Evaluate trained model.

        Args:
            ds_test (tf.data.Dataset): Test dataset.
        """
        self.model.evaluate(ds_test)
