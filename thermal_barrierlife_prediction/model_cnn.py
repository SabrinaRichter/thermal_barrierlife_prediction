import tensorflow as tf
from typing import Tuple


class ModelCNN:
    
    def __init__(
            self,
    ):
        """
        """

        input_x = tf.keras.layers.Input(
            ...
        )

        output = ... # tf model stuff


        self.training_model = tf.keras.models.Model(
            inputs=[input_x],
            outputs=output,
            name='basic_cnn'
        )
