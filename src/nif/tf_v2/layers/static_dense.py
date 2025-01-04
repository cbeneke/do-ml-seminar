# StaticDense is a Layer that implements a dense layer with given weights and biases

import tensorflow as tf

class StaticDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, weights_indexes=None, biases_indexes=None, input_dim=None, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.weights_indexes = weights_indexes
        self.biases_indexes = biases_indexes
        self.input_dim = input_dim

    def call(self, inputs):
        # Split input into data and parameters
        x = inputs[:, :self.input_dim]
        parameters = inputs[:, self.input_dim:]
        
        # Extract weights and biases
        weights = tf.reshape(
            parameters[:, self.weights_indexes[0]:self.weights_indexes[1]], 
            (-1, self.input_dim, self.units)
        )
        biases = parameters[:, self.biases_indexes[0]:self.biases_indexes[1]]
        
        # Compute output activation(x * weights + biases)
        output = tf.matmul(tf.expand_dims(x, 1), weights)
        output = tf.squeeze(output, axis=1) + biases
    
        if self.activation is not None:
            output = self.activation(output)

        output = tf.concat([output, parameters], axis=1)
        return output