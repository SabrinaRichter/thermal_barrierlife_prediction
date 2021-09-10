import tensorflow as tf


class Convolution(tf.keras.layers.Layer):
    """
    Convolution stack with different layer types
    """

    def __init__(
            self,
            layer_type: str,
            filters: list,
            activation: str,
            kernel_size: list,
            strides: list,
            pool_size: list,
            pool_strides: list,
            padding='same',
            dtype="float32",
            **kwargs
    ):

        super(Convolution, self).__init__(dtype=dtype, **kwargs)
        self.fwd_pass = []

        for i in range(len(filters)):
            if layer_type == 'mix_pool':
                self.fwd_pass.append(ConvolutionMixedPooling(
                    filters=filters[i],
                    activation=activation,
                    kernel_size=kernel_size[i],
                    strides=strides[i],
                    max_pool_pool_size=pool_size[i],
                    max_pool_strides=pool_strides[i],
                    avg_pool_pool_size=pool_size[i],
                    avg_pool_strides=pool_strides[i],
                    padding=padding,
                    dtype=dtype,
                ))
            elif layer_type == 'max_pool':
                self.fwd_pass.append(ConvolutionPooling(
                    filters=filters[i],
                    activation=activation,
                    kernel_size=kernel_size[i],
                    strides=strides[i],
                    pool_type='max',
                    pool_size=pool_size[i],
                    pool_strides=pool_strides[i],
                    padding=padding,
                    dtype="float32",
                ))
            else:
                raise ValueError('Layer type unrecognised')
        self.fwd_pass.append(tf.keras.layers.Flatten())

    def call(self, inputs, **kwargs):
        x = inputs
        #x = tf.expand_dims(x, axis=3)
        for layer in self.fwd_pass:
            x = layer(x)
        return x


class ConvolutionMixedPooling(tf.keras.layers.Layer):
    """
    Convolution layer that mixes max and avg pooling with learned weight
    """

    def __init__(
            self,
            filters: int,
            activation: str,
            kernel_size: tuple,
            strides: tuple,
            max_pool_pool_size: tuple,
            max_pool_strides: tuple,
            avg_pool_pool_size: tuple,
            avg_pool_strides: tuple,
            padding: str,
            dtype="float32",
            **kwargs
    ):
        super(ConvolutionMixedPooling, self).__init__(dtype=dtype, **kwargs)

        self.conv = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides, activation=activation, padding=padding,
        )
        self.avgpool = tf.keras.layers.AveragePooling2D(
            pool_size=avg_pool_pool_size, strides=avg_pool_strides,
        )
        self.maxpool = tf.keras.layers.MaxPool2D(
            pool_size=max_pool_pool_size, strides=max_pool_strides,
        )
        self.mix = MixChannels(n_channels=2, dtype=dtype)

    def call(self, inputs, **kwargs):
        x = inputs
        x_conv = self.conv(x)
        x_maxpool = self.maxpool(x_conv)
        x_avgpool = self.avgpool(x_conv)
        x = self.mix([x_avgpool, x_maxpool])
        return x


class ConvolutionPooling(tf.keras.layers.Layer):
    """
    Convolution layer with max or abs pooling
    """

    def __init__(
            self,
            filters: int,
            activation: str,
            kernel_size: tuple,
            strides: tuple,
            pool_type: str,
            pool_size: tuple,
            pool_strides: tuple,
            padding: str,
            dtype="float32",
            **kwargs
    ):
        """
        :param pool_type: max or avg
        """
        super(ConvolutionPooling, self).__init__(dtype=dtype, **kwargs)

        self.conv = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides, activation=activation, padding=padding
        )
        if pool_type == 'avg':
            self.pool = tf.keras.layers.AveragePooling2D(
                pool_size=pool_size, strides=pool_strides,
            )
        elif pool_type == 'max':
            self.pool = tf.keras.layers.MaxPool2D(
                pool_size=pool_size, strides=pool_strides,
            )
        else:
            raise ValueError('Pooling operation not recognised')

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.conv(x)
        x = self.pool(x)
        return x


class MixChannels(tf.keras.layers.Layer):
    """
    Weighted sum of multiple matrices of same size
    """

    def __init__(
            self,
            n_channels: int,
            dtype="float32",
            **kwargs
    ):
        """

        """
        super(MixChannels, self).__init__(dtype=dtype, **kwargs)
        self.ws = self.add_weight(
            shape=[n_channels],
            initializer="ones",
            dtype=dtype
        )

    def call(self, inputs, **kwargs):
        # Normalize weights
        ws = tf.math.divide(self.ws, tf.reduce_sum(self.ws))
        # Multiply each channel with weight and sum channels
        x = None
        for i, channel in enumerate(inputs):
            # Do not allow zero weights (would loose output from that point on)
            w = ws[i]
            if w == 0:
                w = 1e-16
            channel = tf.math.scalar_mul(
                w, channel
            )
            if x is None:
                x = channel
            else:
                x = x + channel
        return x


# UNIMPLEMENTED!!!! TODO
# class ConvolutionChannels(tf.keras.layers.Layer):
#
#     def __init__(
#             self,
#             n_layers: int,
#             filters: int,
#             kernel_size: tuple,
#             strides: tuple,
#             max_pool: bool,
#             avg_pool: bool,
#             max_pool_pool_size: tuple,
#             max_pool_strides: tuple,
#             avg_pool_pool_size: tuple,
#             avg_pool_strides: tuple,
#             dtype="float32",
#             **kwargs
#     ):
#         raise NotImplementedError('Not implemented')
#
#         super(ConvolutionChannels, self).__init__(dtype=dtype, **kwargs)
#         self.fwd_pass_channels = {}
#         # If using pooling make separate convolution layers for each each pooling type
#         if max_pool:
#             self.fwd_pass_channels['max_pool'] = []
#         if avg_pool:
#             self.fwd_pass_channels['avg_pool'] = []
#         # If no pooling is used use only convolution layers
#         if len(self.fwd_pass_channels) < 0:
#             self.fwd_pass_channels['bare_conv'] = []
#
#         for i in range(n_layers):
#             if 'max_pool' in self.fwd_pass_channels:
#                 self.fwd_pass_channels['max_pool'].append(
#                     tf.keras.layers.Conv2D(
#                         filters=filters, kernel_size=kernel_size, strides=strides,
#                     ))
#                 self.fwd_pass_channels['max_pool'].append(tf.keras.layers.MaxPool2D(
#                     pool_size=max_pool_pool_size, strides=max_pool_strides,
#                 ))
#             if 'avg_pool' in self.fwd_pass_channels:
#                 self.fwd_pass_channels['avg_pool'].append(
#                     tf.keras.layers.Conv2D(
#                         filters=filters, kernel_size=kernel_size, strides=strides,
#                     ))
#                 self.fwd_pass_channels['avg_pool'].append(tf.keras.layers.AveragePool2D(
#                     pool_size=avg_pool_pool_size, strides=avg_pool_strides,
#                 ))
#             if 'bare_conv' in self.fwd_pass_channels:
#                 self.fwd_pass_channels['bare_conv'].append(
#                     tf.keras.layers.Conv2D(
#                         filters=filters, kernel_size=kernel_size, strides=strides,
#                     ))
#
#     def call(self, inputs, **kwargs):
#         # IDea pass through all channels and concat
#         x_all = []
#         for channel, layers in self.fwd_pass_channels.items():
#             x = inputs


class DenseEncoderDecoder(tf.keras.layers.Layer):
    """
    Dense stack
    """

    def __init__(
            self,
            units,
            l1_coef: float,
            l2_coef: float,
            activation: str,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            regularize: bool = False,
            norm: bool = False,
            dropout_rate: float = 0.0,
            dtype="float32",
            **kwargs
    ):
        """


        """
        super(DenseEncoderDecoder, self).__init__(dtype=dtype, **kwargs)
        self.fwd_pass = []
        for i, x in enumerate(units):
            self.fwd_pass.append(
                tf.keras.layers.Dense(
                    units=x,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_coef, l2=l2_coef) if regularize else None,
                    bias_regularizer=tf.keras.regularizers.l1_l2(l1=l1_coef, l2=l2_coef) if regularize else None,
                )
            )
            # Batch normalisation
            if norm:
                self.fwd_pass.append(
                    tf.keras.layers.LayerNormalization(center=True, scale=True)
                )
            # Dropout
            if dropout_rate > 0.0:
                if activation == "selu":
                    self.fwd_pass.append(tf.keras.layers.AlphaDropout(dropout_rate))
                else:
                    self.fwd_pass.append(tf.keras.layers.Dropout(dropout_rate, noise_shape=None, seed=None))

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.fwd_pass:
            x = layer(x)
        return x


class Prediction(tf.keras.layers.Layer):
    """
    Prediction layer with 1 (e.g. mean) or 2 (e.g. mean and std) outputs
    """

    def __init__(
            self,
            split_output: bool,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation: str = 'linear',
            dtype="float32",
            **kwargs
    ):
        """

        :param split_output: Output 1 or 2 values
        """
        super(Prediction, self).__init__(dtype=dtype, **kwargs)
        self.layer = tf.keras.layers.Dense(
            units=1 if not split_output else 2,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        self.split_output = split_output

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.layer(x)
        if self.split_output:  # Used to generate two moments per output node, e.g. for variational posteriors.
            x = tf.split(x, num_or_size_splits=2, axis=1)
            # Overwrite scale model if scale is constant:
            if self.scale is not None:
                x[1] = tf.zeros_like(x[1]) + self.scale
            x0 = x[0]
            x1 = x[1]
        else:  # Used to generate point estimate per output node, e.g. for deterministic layer stacks.
            x0 = x
            x1 = None

        if x1 is None:
            return x0
        else:
            return x0, x1
