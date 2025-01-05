import tensorflow as tf

from nif.tf_v2.layers.shortcut import Shortcut
from nif.tf_v2.layers.resnet import ResNet
from nif.tf_v2.layers.static_dense import StaticDense

class NIF(tf.keras.Model):
    def __init__(self, cfg_shape_net, cfg_parameter_net, dtype="float32"):
        super().__init__(dtype=dtype)

        # Store the config
        self.cfg_shape_net = cfg_shape_net
        self.cfg_parameter_net = cfg_parameter_net

        # Setup for standard regularization
        # 1. regularization for kernel in parameter net
        if isinstance(cfg_parameter_net.get("l2_reg", None), (float, int)):
            self.pnet_kernel_regularizer = tf.keras.regularizers.L2(cfg_parameter_net.get("l2_reg", None))
            self.pnet_bias_regularizer = tf.keras.regularizers.L2(cfg_parameter_net.get("l2_reg", None))
        elif isinstance(cfg_parameter_net.get("l1_reg", None), (float, int)):
            self.pnet_kernel_regularizer = tf.keras.regularizers.L1(cfg_parameter_net.get("l1_reg", None))
            self.pnet_bias_regularizer = tf.keras.regularizers.L1(cfg_parameter_net.get("l1_reg", None))
        else:
            self.pnet_kernel_regularizer = None
            self.pnet_bias_regularizer = None

        # 2. output of parameter net regularization
        if isinstance(cfg_parameter_net.get("act_l2_reg", None), (float, int)):
            self.pnet_act_regularizer = tf.keras.regularizers.L2(cfg_parameter_net.get("act_l2_reg", None))
        elif isinstance(cfg_parameter_net.get("act_l1_reg", None), (float, int)):
            self.pnet_act_regularizer = tf.keras.regularizers.L1(cfg_parameter_net.get("act_l1_reg", None))
        else:
            self.pnet_act_regularizer = None

        # Build parameter net
        self.parameter_net_layers = self._build_parameter_net(
            self.cfg_parameter_net,
            self.cfg_shape_net,
            self.dtype,
            self.pnet_kernel_regularizer,
            self.pnet_bias_regularizer,
            self.pnet_act_regularizer,
        )

        # Build shape net
        self.shape_net_layers = self._build_shape_net(
            self.cfg_shape_net,
            self.dtype,
        )

    @staticmethod
    def _build_parameter_net(cfg_parameter_net, cfg_shape_net, dtype, kernel_regularizer, bias_regularizer, activity_regularizer) -> list:
        output_dim = NIF._get_parameter_net_output_dim(cfg_shape_net, cfg_parameter_net)
        layers = []

        # First Layer
        layers.append(tf.keras.layers.Dense(
            cfg_parameter_net["units"],
            activation=tf.keras.activations.get(cfg_parameter_net["activation"]),
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dtype=dtype,
        ))

        # Hidden Layers - ResNet or Shortcut based on config
        HiddenLayer = ResNet if cfg_parameter_net.get("use_resblock", False) else Shortcut
        for _ in range(cfg_parameter_net["nlayers"] - 1):
            layers.append(HiddenLayer(
                units=cfg_parameter_net["units"],
                activation=cfg_parameter_net["activation"],
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                dtype=dtype,
            ))
        
        # Last Layer with additional activation regularizer
        layers.append(tf.keras.layers.Dense(
            output_dim,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            dtype=dtype,
        ))
        return layers

    @staticmethod
    def _build_shape_net(cfg_shape_net, dtype) -> list:
        layers = []

        # First layer -- input_dim -> units fully connected
        layers.append(StaticDense(
            units=cfg_shape_net["units"],
            activation=cfg_shape_net["activation"],
            weights_indexes=[0, cfg_shape_net["input_dim"] * cfg_shape_net["units"]],
            biases_indexes=[0, cfg_shape_net["units"]],
            input_dim=cfg_shape_net["input_dim"],
            dtype=dtype,
        ))
        for i in range(cfg_shape_net["nlayers"]):
            # Hidden layer -- units -> units fully connected
            layers.append(StaticDense(
                units=cfg_shape_net["units"],
                activation=cfg_shape_net["activation"],
                weights_indexes=[
                    cfg_shape_net["input_dim"] * cfg_shape_net["units"] + i * cfg_shape_net["units"]**2,
                    cfg_shape_net["input_dim"] * cfg_shape_net["units"] + (i + 1) * cfg_shape_net["units"]**2
                ],
                biases_indexes=[
                    (i+1) * cfg_shape_net["units"],
                    (i+2) * cfg_shape_net["units"]
                ],
                input_dim=cfg_shape_net["units"],
                dtype=dtype,
            ))
        # Last layer -- units -> output_dim fully connected
        layers.append(StaticDense(
            units=cfg_shape_net["output_dim"],
            activation=cfg_shape_net["activation"],
            weights_indexes=[
                cfg_shape_net["input_dim"] * cfg_shape_net["units"] + cfg_shape_net["nlayers"] * cfg_shape_net["units"]**2,
                cfg_shape_net["input_dim"] * cfg_shape_net["units"] + cfg_shape_net["nlayers"] * cfg_shape_net["units"]**2 + cfg_shape_net["output_dim"] * cfg_shape_net["units"]
            ],
            biases_indexes=[
                cfg_shape_net["nlayers"] * cfg_shape_net["units"],
                cfg_shape_net["nlayers"] * cfg_shape_net["units"] + cfg_shape_net["output_dim"]
            ],
            input_dim=cfg_shape_net["units"],
            dtype=dtype,
        ))
        return layers

    @staticmethod
    def _get_parameter_net_output_dim(cfg_shape_net, cfg_parameter_net):
        return (
            cfg_shape_net["nlayers"] * cfg_shape_net["units"]**2  # Each hidden layer has a units*units weight matrix
            + (cfg_parameter_net["input_dim"] + cfg_shape_net["output_dim"] + 1 + cfg_shape_net["nlayers"]) * cfg_shape_net["units"]  # First layer + hidden layers + output layer have "units" biases
            + cfg_shape_net["output_dim"]  # Output layer has "output_dim" weights
        )

    def call(self, inputs, training=None, mask=None):
        # Call parameter net
        parameter_net_output = inputs[:, :self.cfg_parameter_net["input_dim"]]
        for layer in self.parameter_net_layers:
            parameter_net_output = layer(parameter_net_output)

        # Call shape net
        shape_net_output = tf.concat([
            inputs[:, :self.cfg_parameter_net["input_dim"]],
            parameter_net_output
        ], axis=1)
        for layer in self.shape_net_layers:
            shape_net_output = layer(shape_net_output)

        # Drop weights and biases from shape net output
        output = shape_net_output[:, :self.cfg_shape_net["output_dim"]]

        return output

    def build(self, input_shape):
        # Build parameter net layers
        x_shape = (None, self.cfg_parameter_net["input_dim"])
        for layer in self.parameter_net_layers:
            layer.build(x_shape)
            x_shape = layer.compute_output_shape(x_shape)

        # Build shape net layers
        # Input shape includes both original input and parameter net output
        x_shape = (None, self.cfg_parameter_net["input_dim"] + self._get_parameter_net_output_dim(self.cfg_shape_net, self.cfg_parameter_net))
        for layer in self.shape_net_layers:
            layer.build(x_shape)
            x_shape = layer.compute_output_shape(x_shape)

        self.built = True
        super().build(input_shape)