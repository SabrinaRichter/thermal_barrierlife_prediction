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
        # x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation="relu", input_shape=(32, 32, 3))(x)
        # x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
        #
        # x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation="relu")(x)
        # x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(100, activation="relu")(x)
        output = tf.keras.layers.Dense(1, activation="linear")(x)


        self.training_model = tf.keras.models.Model(
            inputs=[input_x],
            outputs=output,
            name='basic_cnn'
        )
