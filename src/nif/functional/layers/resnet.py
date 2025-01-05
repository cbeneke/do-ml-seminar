# Implements a ResNet-like layer with residual connections

import tensorflow as tf

class ResNet(tf.keras.layers.Layer):
    def __init__(self, units, activation, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, dtype, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.dtype = dtype

        self.dense_activation = tf.keras.layers.Dense(
            self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            dtype=self.dtype,
        )

        self.dense_no_activation = tf.keras.layers.Dense(
            self.units,
            activation=None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            dtype=self.dtype,
        )

    def call(self, inputs):
        x = self.dense_activation(inputs)
        x = self.dense_no_activation(x)
        output = self.activation(tf.keras.layers.add([inputs, x]))
        return output


