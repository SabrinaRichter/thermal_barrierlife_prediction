import tensorflow as tf
from tensorflow.keras import layers
from thermal_barrierlife_prediction.layers import Convolution, DenseEncoderDecoder, Prediction
from FR import Fourier_Transformation
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# import keras

class ModelCNN:

    def __init__(
            self,
            input_shapes,
            layer_type='mix_pool',
            filters=[2, 4, 8, 16],
            kernel_size=[2, 2, 2, 2],
            strides=[1, 1, 1, 1],
            pool_size=[2, 2, 2, 2],
            pool_strides=[2, 2, 2, 2],
            padding='same',
            activation_conv='relu',
            units_dense=[100],
            init_kernel_dense='glorot_uniform',
            init_bias_dense='zeros',
            regularize_dense=False,
            l1_dense=0.0,
            l2_dense=0.0,
            activation_dense='relu',
            init_kernel_pred='glorot_uniform',
            init_bias_pred='zeros',
            activation_pred='linear',
            FGF_guassian_projection=16,
            FGF_scale=10,
            crop_image_size=256,
            split_output=False,
            use_cov: bool = False,
    ):
        """
        Creates the tf model.
        :param layer_type: Convolution layer type: mix_pool or max_pool
        """
        # data_augmentation = keras.Sequential([keras.layers.experimental.preprocessing.RandomCrop(height=512,width=512,seed=42)])
        if len(filters) != len(kernel_size) != strides != len(pool_size) != len(pool_strides):
            raise ValueError('Filter, kernel, pool, and stride lists should have same legnth')
        # data_augmentation = keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomContrast(factor=(0.1, 2))])

        input_x = tf.keras.layers.Input(
            shape=input_shapes[0],
        )
        input_x_cov = tf.keras.layers.Input(
            shape=input_shapes[1],
        )
        x = input_x
        x = tf.expand_dims(x, -1)
        x = Fourier_Transformation.FourierFeatureProjection(
            gaussian_projection=FGF_guassian_projection,
            gaussian_scale=FGF_scale)(x)

        x = tf.keras.layers.experimental.preprocessing.RandomCrop(height=crop_image_size, width=crop_image_size,
                                                                  seed=42)(x, training=True)

        # Pass through layer stacks
        x = Convolution(
            layer_type=layer_type,
            filters=filters,
            activation=activation_conv,
            kernel_size=kernel_size,
            strides=strides,
            pool_size=pool_size,
            pool_strides=pool_strides,
            padding=padding)(x)

        if use_cov:
            x = tf.keras.layers.Concatenate(axis=1)((x, input_x_cov))

        x = DenseEncoderDecoder(
            units=units_dense,
            kernel_initializer=init_kernel_dense,
            bias_initializer=init_bias_dense,
            regularize=regularize_dense,
            l1_coef=l1_dense,
            l2_coef=l2_dense,
            activation=activation_dense)(x)

        output = Prediction(
            kernel_initializer=init_kernel_pred,
            bias_initializer=init_bias_pred,
            activation=activation_pred,
            split_output=split_output)(x)

        # Training model
        self.training_model = tf.keras.models.Model(
            inputs=[input_x, input_x_cov],
            outputs=output,
            name='basic_cnn'
        )
