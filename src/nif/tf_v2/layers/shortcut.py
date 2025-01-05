# Implements a ResNet-like layer with residual connections

import tensorflow as tf

class Shortcut(tf.keras.layers.Layer):
    def __init__(self, units, activation, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.units = units

        self.dense = tf.keras.layers.Dense(
            self.units,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dtype=dtype,
        )

    def call(self, inputs):
        x = self.dense(inputs)
        output = tf.keras.layers.add([inputs, x])
        return output

    def compute_output_shape(self, input_shape):
        # Since this is a residual connection, output shape is same as input
        return input_shape
