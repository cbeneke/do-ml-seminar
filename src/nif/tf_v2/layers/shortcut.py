# Implements a ResNet-like layer with residual connections

import tensorflow as tf

class Shortcut(tf.keras.layers.Layer):
    def __init__(self, units, activation, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.units = units
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.dense = tf.keras.layers.Dense(
            self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            dtype=self.dtype,
        )

    def call(self, inputs):
        x = self.dense(inputs)
        output = tf.keras.layers.add([inputs, x])
        return output
