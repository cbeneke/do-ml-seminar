import tensorflow as tf
from nif.functional.layers.hyper_layer import HyperLayer

# HyperSIREN implements a SIREN layer with given weights and biases
class HyperSIREN(HyperLayer):
    def __init__(self, units, omega_0, activation, **kwargs):
        super().__init__(**kwargs)

        self.units = units
        self.omega_0 = omega_0
        self.activation = activation
    @tf.function
    def call_step(self, inputs, weights, biases):
        x = tf.einsum("ai,aij->aj", inputs, weights)
        if self.activation == 'sine':
            x = self.omega_0 * x
            x = tf.add(x, biases)
            x = tf.math.sin(x)
        
        return x