import tensorflow as tf


class ModelCNN:
    
    def __init__(
            self,
            input_shape,
    ):
        """
        Creates the tf model.
        """

        input_x = tf.keras.layers.Input(
            shape=input_shape,
        )

        x = input_x
        x = tf.expand_dims(x, 3)
        x = tf.keras.layers.Conv2D(4, (3, 3), strides=(2, 2), padding='same', activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)

        x = tf.keras.layers.Conv2D(16, (5, 5), padding='same', activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)

        x = tf.keras.layers.Conv2D(32, (5, 5), strides=(5, 5),  padding='same', activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)

        x = tf.keras.layers.Conv2D(64, (5, 5), strides=(5, 5), padding='same', activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(100, activation="relu")(x)
        output = tf.keras.layers.Dense(1, activation="linear")(x)


        self.training_model = tf.keras.models.Model(
            inputs=[input_x],
            outputs=output,
            name='basic_cnn'
        )
