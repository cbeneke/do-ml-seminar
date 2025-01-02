# Implements a ResNet-like layer with residual connections

import tensorflow as tf

class ResNet(tf.keras.layers.Layer):
    def __init__(self, units, activation, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, mixed_policy, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.mixed_policy = mixed_policy

    def call(self, inputs):
        x = inputs
        y = tf.keras.layers.Dense(
            self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            dtype=self.mixed_policy,
        )(x)
        y = tf.keras.layers.Dense(
            self.units,
            activation=None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            dtype=self.mixed_policy,
        )(y)
        output = self.activation(tf.keras.layers.add([x, y]))
        return output
