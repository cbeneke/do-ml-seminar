# StaticDense is a Layer that implements a dense layer with given weights and biases

import tensorflow as tf

class StaticDense(tf.keras.layers.Layer):
    def __init__(self, units, activation, weights_indexes, biases_indexes, dtype=None,**kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.weights_indexes = weights_indexes
        self.biases_indexes = biases_indexes

    @tf.function
    def call(self, inputs):
        x = inputs[0]
        weights = inputs[1]
        biases = inputs[2]
        x_weights = tf.reshape(weights[:, self.weights_indexes], [-1, self.weights_indexes.shape[0], self.weights_indexes.shape[1]])
        x_biases = tf.reshape(biases[:, self.biases_indexes], [-1, self.biases_indexes.shape[0]])

        return self.activation(tf.matmul(x, x_weights) + x_biases), weights, biases

    @tf.function
    def compute_output_spec(self, input_spec):
        output_dim = self.units + self.weights_indexes[1] - self.weights_indexes[0] + self.biases_indexes[1] - self.biases_indexes[0]
        return tf.TensorSpec(shape=(None, output_dim), dtype=self.dtype)
