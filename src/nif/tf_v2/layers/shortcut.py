# Implements a ResNet-like layer with residual connections

import tensorflow as tf

class Shortcut(tf.keras.layers.Layer):
    def __init__(self, units, activation, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, dtype=None, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            units,
            use_bias=True,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dtype=dtype,
        )

    @property
    def trainable_weights(self):
        # Expose the dense layer's trainable weights
        return self.dense.trainable_weights

    @property 
    def non_trainable_weights(self):
        # Expose the dense layer's non-trainable weights
        return self.dense.non_trainable_weights

    def build(self, input_shape):
    # Build the dense layer with correct input shape
        self.dense.build(input_shape)
        self.built = True

    def call(self, inputs, training=False):
        x = self.dense(inputs, training=training)
        x += inputs
        return x

    def compute_output_shape(self, input_shape):
        # Since this is a residual connection, output shape is same as input
        return input_shape