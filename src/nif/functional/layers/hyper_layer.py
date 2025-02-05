import tensorflow as tf

# HyperLayer is a base class for layers that utilise provided weights and biases
class HyperLayer(tf.keras.layers.Layer):
    def __init__(self, weights_from, weights_to, bias_offset, biases_from, biases_to, units, activation, **kwargs):
        super().__init__(**kwargs)

        self.weights_from = weights_from
        self.weights_to = weights_to

        self.biases_from = bias_offset + biases_from
        self.biases_to = bias_offset + biases_to

        self.units = units
        self.activation = activation

        self.built = False

    def build(self, input_shape):
        self._input_dim = input_shape[-1]
        self.built = True

    @tf.function
    def _parse_parameters(self, parameters):
        weights = parameters[:, self.weights_from:self.weights_to]
        weights = tf.reshape(weights, (-1, self._input_dim, self.units))

        biases = parameters[:, self.biases_from:self.biases_to]
        biases = tf.reshape(biases, (-1, self.units))

        return weights, biases

    @tf.function
    def call_step(self, inputs, weights, biases):
        raise NotImplementedError("call_step must be implemented when using HyperLayer")

    @tf.function
    def call(self, inputs):
        inputs, parameters = inputs
        weights, biases = self._parse_parameters(parameters)

        x = self.call_step(inputs, weights, biases)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)