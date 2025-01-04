# StaticDense is a Layer that implements a dense layer with given weights and biases

import tensorflow as tf

class StaticDense(tf.keras.layers.Layer):
    def __init__(self, units, activation, weights_indexes, biases_indexes, input_dim, dtype=None,**kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.weights_indexes = weights_indexes
        self.biases_indexes = biases_indexes
        self.input_dim = input_dim

    @tf.function
    def call(self, inputs):
        input = inputs[:, :self.units]
        weights = tf.reshape(inputs[:, self.weights_indexes[0]:self.weights_indexes[1]], [-1, self.weights_indexes[0], self.weights_indexes[1]])
        biases = tf.reshape(inputs[:, self.biases_indexes[0]:self.biases_indexes[1]], [-1, self.biases_indexes[0], self.biases_indexes[1]])
       
        output = tf.zeros(shape=(inputs.shape[0], inputs.shape[1] - self.input_dim + self.units ), dtype=self.dtype)
        output[:, :self.units] = self.activation(tf.matmul(input, weights) + biases)
        output[:, self.units:] = inputs[:, self.input_dim:]

        return output

    @tf.function
    def compute_output_spec(self, input_spec):
        output_dim = self.units + self.weights_indexes[1] - self.weights_indexes[0] + self.biases_indexes[1] - self.biases_indexes[0]
        return tf.TensorSpec(shape=(None, output_dim), dtype=self.dtype)
