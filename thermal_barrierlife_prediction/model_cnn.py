import tensorflow as tf

from thermal_barrierlife_prediction.layers import ConvolutionMixedPooling, DenseEncoderDecoder, Prediction


class ModelCNN:

    def __init__(
            self,
            input_shape,
            n_conv=1,
            #filters=1,
            kernel_size=2,
            strides=2,
            max_pool_pool_size=2,
            max_pool_strides=2,
            avg_pool_pool_size=2,
            avg_pool_strides=2,
            n_dense=1,
            init_dense='ones',
            l1_dense=0.0,
            l2_dense=0.0,
            activation_dense='linear',
            init_pred='ones',
            activation_pred='linear',
            split_output=False
    ):
        """
        Creates the tf model.
        """

        input_x = tf.keras.layers.Input(
            shape=input_shape,
        )
        x = input_x
        x = ConvolutionMixedPooling(
            n_layers=n_conv,
            filters=1,
            kernel_size=kernel_size,
            strides=strides,
            max_pool_pool_size=max_pool_pool_size,
            max_pool_strides=max_pool_strides,
            avg_pool_pool_size=avg_pool_pool_size,
            avg_pool_strides=avg_pool_strides)(x)
        x = DenseEncoderDecoder(
            units=n_dense,
            initializer=init_dense,
            l1_coef=l1_dense,
            l2_coef=l2_dense,
            activation=activation_dense)(x)
        output = Prediction(
            initializer=init_pred,
            activation=activation_pred,
            split_output=split_output)(x)
        self.training_model = tf.keras.models.Model(
            inputs=[input_x],
            outputs=output,
            name='basic_cnn'
        )
