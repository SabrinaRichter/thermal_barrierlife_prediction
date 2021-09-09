import tensorflow as tf


class ConvolutionMixedPooling(tf.keras.layers.Layer):
    def __init__(
            self,
            n_layers: int,
            filters: int,
            kernel_size: tuple,
            strides: tuple,
            max_pool_pool_size: tuple,
            max_pool_strides: tuple,
            avg_pool_pool_size: tuple,
            avg_pool_strides: tuple,
            dtype="float32",
            **kwargs
    ):

        super(ConvolutionMixedPooling, self).__init__(dtype=dtype, **kwargs)
        self.fwd_pass_conv = []
        self.fwd_pass_maxpool = []
        self.fwd_pass_avgpool = []
        self.fwd_pass_mix = []
        self.n_layers = n_layers

        for i in range(self.n_layers):
            self.fwd_pass_conv.append(
                tf.keras.layers.Conv2D(
                    filters=filters, kernel_size=kernel_size, strides=strides,
                ))
            self.fwd_pass_avgpool.append(tf.keras.layers.AveragePool2D(
                pool_size=avg_pool_pool_size, strides=avg_pool_strides,
            ))
            self.fwd_pass_maxpool.append(tf.keras.layers.MaxPool2D(
                pool_size=max_pool_pool_size, strides=max_pool_strides,
            ))
            self.fwd_pass_mix.append(MixChannels(n_channels=2, dtype=dtype))
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, **kwargs):
        x = inputs
        for i in range(self.n_layers):
            x_conv = self.fwd_pass_conv[i](x)
            x_maxpool = self.fwd_pass_maxpool[i](x_conv)
            x_avgpool = self.fwd_pass_avgpool[i](x_conv)
            x = self.fwd_pass_mix[i]([x_avgpool, x_maxpool])
        x = self.flatten(x)
        return x


class MixChannels(tf.keras.layers.Layer):
    """

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
            channel = tf.math.scalar_mul(
                ws[i], channel
            )
            if x is None:
                x = channel
            else:
                x = x + channel
        return x


# UNIMPLEMENTED!!!! TODO
class ConvolutionChannels(tf.keras.layers.Layer):

    def __init__(
            self,
            n_layers: int,
            filters: int,
            kernel_size: tuple,
            strides: tuple,
            max_pool: bool,
            avg_pool: bool,
            max_pool_pool_size: tuple,
            max_pool_strides: tuple,
            avg_pool_pool_size: tuple,
            avg_pool_strides: tuple,
            dtype="float32",
            **kwargs
    ):
        raise NotImplementedError('Not implemented')

        super(ConvolutionChannels, self).__init__(dtype=dtype, **kwargs)
        self.fwd_pass_channels = {}
        # If using pooling make separate convolution layers for each each pooling type
        if max_pool:
            self.fwd_pass_channels['max_pool'] = []
        if avg_pool:
            self.fwd_pass_channels['avg_pool'] = []
        # If no pooling is used use only convolution layers
        if len(self.fwd_pass_channels) < 0:
            self.fwd_pass_channels['bare_conv'] = []

        for i in range(n_layers):
            if 'max_pool' in self.fwd_pass_channels:
                self.fwd_pass_channels['max_pool'].append(
                    tf.keras.layers.Conv2D(
                        filters=filters, kernel_size=kernel_size, strides=strides,
                    ))
                self.fwd_pass_channels['max_pool'].append(tf.keras.layers.MaxPool2D(
                    pool_size=max_pool_pool_size, strides=max_pool_strides,
                ))
            if 'avg_pool' in self.fwd_pass_channels:
                self.fwd_pass_channels['avg_pool'].append(
                    tf.keras.layers.Conv2D(
                        filters=filters, kernel_size=kernel_size, strides=strides,
                    ))
                self.fwd_pass_channels['avg_pool'].append(tf.keras.layers.AveragePool2D(
                    pool_size=avg_pool_pool_size, strides=avg_pool_strides,
                ))
            if 'bare_conv' in self.fwd_pass_channels:
                self.fwd_pass_channels['bare_conv'].append(
                    tf.keras.layers.Conv2D(
                        filters=filters, kernel_size=kernel_size, strides=strides,
                    ))

    def call(self, inputs, **kwargs):
        # IDea pass through all channels and concat
        x_all = []
        for channel, layers in self.fwd_pass_channels.items():
            x = inputs


class DenseEncoderDecoder(tf.keras.layers.Layer):
    """
    Generator consisting of dense layers.

    Maps from one domain to another: RNA -> metabolites or metabolites -> RNA in this model class.
    """

    def __init__(
            self,
            units,
            initializer,
            l1_coef: float,
            l2_coef: float,
            activation: str,
            norm: bool = False,
            dropout_rate: float = 0.0,
            dtype="float32",
            **kwargs
    ):
        """

        :param units:
        :param initializer:
        :param l1_coef:
        :param l2_coef:
        :param norm:
        :param activation: Activation function to use inside of stack. Used in last layer if inv_linker_loc is None.
        :param dropout_rate:
        :param inv_linker_loc: Output transformation of mean (or point estimator) tensor. Leave None to use activation.
        :param inv_linker_scale: Output transformation of mean (or point estimator) tensor.
            Leave None if split_output is False.
        :param split_output: Whether to generate output of twice reqiured length for variational posterior
            (location and scale output).
        :param dtype:
        :param kwargs:
        """
        super(DenseEncoderDecoder, self).__init__(dtype=dtype, **kwargs)
        self.fwd_pass = []
        for i, x in enumerate(units):
            self.fwd_pass.append(
                tf.keras.layers.Dense(
                    units=x,
                    activation=activation,
                    kernel_initializer=initializer,
                    bias_initializer=initializer,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_coef, l2=l2_coef),
                    bias_regularizer=tf.keras.regularizers.l1_l2(l1=l1_coef, l2=l2_coef),
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

    def __init__(
            self,
            initializer,
            activation: str,
            split_output: bool,
            dtype="float32",
            **kwargs
    ):
        """

        :param units:
        :param initializer:
        :param l1_coef:
        :param l2_coef:
        :param norm:
        :param activation: Activation function to use inside of stack. Used in last layer if inv_linker_loc is None.
        :param dropout_rate:
        :param inv_linker_loc: Output transformation of mean (or point estimator) tensor. Leave None to use activation.
        :param inv_linker_scale: Output transformation of mean (or point estimator) tensor.
            Leave None if split_output is False.
        :param split_output: Whether to generate output of twice reqiured length for variational posterior
            (location and scale output).
        :param dtype:
        :param kwargs:
        """
        super(Prediction, self).__init__(dtype=dtype, **kwargs)
        self.fwd_pass = []
        self.fwd_pass.append(
            tf.keras.layers.Dense(
                units=1 if not split_output else 2,
                activation=activation,
                kernel_initializer=initializer,
                bias_initializer=initializer,
            )
        )

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.fwd_pass:
            x = layer(x)
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
