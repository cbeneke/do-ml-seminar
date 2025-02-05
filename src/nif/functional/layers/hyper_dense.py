import tensorflow as tf

from nif.functional.layers.hyper_layer import HyperLayer

# HyperDense implements a Dense layer with given weights and biases
class HyperDense(HyperLayer):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        if self.activation is not None:
            self.activation_fn = tf.keras.activations.get(self.activation)

    @tf.function
    def call_step(self, inputs, weights, biases):
        x = tf.einsum("ai,aij->aj", inputs, weights)
        x = tf.add(x, biases)
        if self.activation is not None:
            x = self.activation_fn(x)
        return x