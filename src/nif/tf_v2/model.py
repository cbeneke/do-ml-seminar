__all__ = ["NIF"]

import tensorflow as tf
from typing import List
class NIF(tf.keras.Model):
    def __init__(self, cfg_shape_net, cfg_parameter_net, mixed_policy="float32"):
        super().__init__()
        self.cfg_shape_net = cfg_shape_net
        self.cfg_parameter_net = cfg_parameter_net
        self.mixed_policy = mixed_policy

        self.parameter_net = self._build_parameter_net(cfg_parameter_net, cfg_shape_net)
        self.shape_net = self._build_shape_net(cfg_shape_net)

    def _build_parameter_net(self, cfg_parameter_net, cfg_shape_net) -> tf.keras.Model:
        layers: List[tf.keras.layers.Layer] = []
        input_dim = cfg_parameter_net["input_dim"]
        units = cfg_parameter_net["units"]
        nlayers = cfg_parameter_net["nlayers"]
        activation = self._get_activation(cfg_parameter_net["activation"])

        layers.append(tf.keras.layers.Dense(units, input_shape=(input_dim,)))
        layers.append(activation)

        for _ in range(nlayers - 1):
            layers.append(tf.keras.layers.Dense(units))
            layers.append(activation)

        layers.append(tf.keras.layers.Dense(cfg_parameter_net["latent_dim"]))

        return tf.keras.Sequential(layers)

    def _build_shape_net(self, cfg_shape_net) -> tf.keras.Model:
        layers: List[tf.keras.layers.Layer] = []
        input_dim = cfg_shape_net["input_dim"]
        units = cfg_shape_net["units"]
        nlayers = cfg_shape_net["nlayers"]
        activation = self._get_activation(cfg_shape_net["activation"])

        layers.append(tf.keras.layers.Dense(units, input_shape=(input_dim,)))
        layers.append(activation)

        for _ in range(nlayers - 1):
            layers.append(tf.keras.layers.Dense(units))
            layers.append(activation)

        layers.append(tf.keras.layers.Dense(cfg_shape_net["output_dim"]))

        return tf.keras.Sequential(layers)

    def call(self, inputs, training=None, mask=None):
        inputs_parameter = inputs[:, :1]
        inputs_x = inputs[:, 1:]

        shape_net_output = self.shape_net(inputs_x)
        parameter_net_output = self.parameter_net(inputs_parameter)

        return shape_net_output + parameter_net_output

