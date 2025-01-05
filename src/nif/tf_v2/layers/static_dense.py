# StaticDense is a Layer that implements a dense layer with given weights and biases

import tensorflow as tf
from nif.tf_v2 import utils

class StaticDense(tf.keras.Layer):
    def __init__(self, units, activation, weights_from, weights_to, bias_offset, biases_from, biases_to, **kwargs):
        super().__init__(**kwargs)

        self.units = units
        self.activation = activation

        self.weights_from = weights_from
        self.weights_to = weights_to

        self.biases_from = bias_offset + biases_from
        self.biases_to = bias_offset + biases_to

        self.built = False

    def build(self, input_shape):
        self._input_dim = input_shape[-1]

        self.built = True

    @tf.function
    def _parse_parameters(self, parameters):
        assert self.built, "Layer is not built"

        weights = parameters[:, self.weights_from:self.weights_to]
        weights = tf.reshape(weights, (-1, self._input_dim, self.units))

        biases = parameters[:, self.biases_from:self.biases_to]
        biases = tf.reshape(biases, (-1, self.units))

        return weights, biases

    @tf.function
    def call(self, inputs):
        inputs, parameters = inputs
        weights, biases = self._parse_parameters(parameters)

        x = tf.einsum("ai,aij->aj", inputs, weights)
        x = tf.add(x, biases)
        if self.activation is not None:
            x = self.activation(x)
        return x


    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)