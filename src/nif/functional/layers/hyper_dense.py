import tensorflow as tf

from nif.functional.layers.hyper_layer import HyperLayer

# HyperDense implements a Dense layer with given weights and biases
class HyperDense(HyperLayer):
    def __init__(self, units, activation, **kwargs):
        super().__init__(**kwargs)

        self.units = units
        self.activation = activation

    @tf.function
    def call_step(self, inputs, weights, biases):
        x = tf.einsum("ai,aij->aj", inputs, weights)
        x = tf.add(x, biases)
        if self.activation is not None:
            x = self.activation(x)
        return x