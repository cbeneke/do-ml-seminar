# StaticDense is a Layer that implements a dense layer with given weights and biases

import tensorflow as tf
from nif.tf_v2 import utils

class StaticDense(tf.keras.Layer):
    def __init__(self, units, cfg_shape_net, weights_from, weights_to, biases_from, biases_to, **kwargs):
        super().__init__(**kwargs)

        self.units = units
        self.activation = tf.keras.activations.get(cfg_shape_net["activation"])

        self.weights_from = weights_from
        self.weights_to = weights_to

        self.biases_from = utils.get_weights_dim(cfg_shape_net) + biases_from
        self.biases_to = utils.get_weights_dim(cfg_shape_net) + biases_to

        self.built = False

    def build(self, input_shape):
        self._input_dim = input_shape[-1]
        self._weights = None
        self._biases = None

        self.built = True

    def pass_parameters(self, parameters):
        assert self.built, "Layer is not built"

        weights = parameters[:, self.weights_from:self.weights_to]
        weights = tf.reshape(weights, (-1, self._input_dim, self.units))
        self._weights = weights

        biases = parameters[:, self.biases_from:self.biases_to]
        biases = tf.reshape(biases, (-1, self.units))
        self._biases = biases

    def call(self, inputs):
        if self._weights is None or self._biases is None:
            raise AttributeError(
                "You must pass parameters before calling the layer."
            )

        x = tf.einsum("ai,aij->aj", inputs, self._weights)
        x = tf.add(x, self._biases)
        if self.activation is not None:
            x = self.activation(x)
        return x


    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)