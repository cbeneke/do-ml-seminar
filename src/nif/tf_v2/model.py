import tensorflow as tf

from nif.tf_v2.layers.shortcut import Shortcut
from nif.tf_v2.layers.resnet import ResNet
from nif.tf_v2.layers.static_dense import StaticDense

class NIF(tf.keras.Model):
    def __init__(self, cfg_shape_net, cfg_parameter_net, mixed_policy="float32"):
        super().__init__()

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

        # Build the networks
        self.parameter_net = self._build_parameter_net(
            cfg_parameter_net,
            cfg_shape_net,
            mixed_policy,
            self.pnet_kernel_regularizer,
            self.pnet_bias_regularizer,
            self.pnet_act_regularizer,
        )
        self.shape_net = self._build_shape_net(cfg_shape_net, mixed_policy)

    @staticmethod
    def _build_parameter_net(cfg_parameter_net, cfg_shape_net, mixed_policy, kernel_regularizer, bias_regularizer, activity_regularizer) -> tf.keras.Model:
        output_dim = (
            cfg_shape_net["nlayers"] * cfg_shape_net["units"]**2  # Each hidden layer has a units*units weight matrix
            + (cfg_parameter_net["input_dim"] + cfg_shape_net["output_dim"] + 1 + cfg_shape_net["nlayers"]) * cfg_shape_net["units"]  # First layer + hidden layers + output layer have "units" biases
            + cfg_shape_net["output_dim"]  # Output layer has "output_dim" weights
        )
        model = tf.keras.Sequential()

        # First Layer
        model.add(tf.keras.layers.Dense(
            cfg_parameter_net["units"],
            activation=tf.keras.activations.get(cfg_parameter_net["activation"]),
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dtype=mixed_policy,
        ))

        # Hidden Layers - ResNet or Shortcut based on config
        HiddenLayer = ResNet if cfg_parameter_net.get("use_resblock", False) else Shortcut
        for _ in range(cfg_parameter_net["nlayers"] - 1):
            model.add(HiddenLayer(
                units=cfg_parameter_net["units"],
                activation=cfg_parameter_net["activation"],
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                mixed_policy=mixed_policy,
            ))
        
        # Last Layer with additional activation regularizer
        model.add(tf.keras.layers.Dense(
            output_dim,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            dtype=mixed_policy,
        ))
        return model

    @staticmethod
    def _build_shape_net(cfg_shape_net, mixed_policy) -> tf.keras.Model:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=1))

        # First layer -- 0 : si_dim * n_sx
        model.add(StaticDense(
            units=cfg_shape_net["units"],
            activation=cfg_shape_net["activation"],
            weights_indexes=[0, cfg_shape_net["input_dim"] * cfg_shape_net["units"]],
            biases_indexes=[0, cfg_shape_net["units"]],
            dtype=mixed_policy,
        ))
        # Hidden layers -- si_dim * n_sx + i * n_sx**2 : si_dim * n_sx + (i + 1) * n_sx**2
        for i in range(cfg_shape_net["nlayers"]):
            model.add(StaticDense(
                units=cfg_shape_net["units"],
                activation=cfg_shape_net["activation"],
                weights_indexes=[
                    cfg_shape_net["input_dim"] * cfg_shape_net["units"] + i * cfg_shape_net["units"]**2,
                    cfg_shape_net["input_dim"] * cfg_shape_net["units"] + (i + 1) * cfg_shape_net["units"]**2
                ],
                biases_indexes=[
                    i * cfg_shape_net["units"],
                    (i+1) * cfg_shape_net["units"]
                ],
                dtype=mixed_policy,
            ))
        # Last layer -- si_dim * n_sx + l_sx * n_sx**2 : si_dim * n_sx + l_sx * n_sx**2 + so_dim * n_sx
        model.add(StaticDense(
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
            dtype=mixed_policy,
        ))
        return model

    def call(self, inputs, training=None, mask=None):
        inputs_parameter = inputs[:, :self.cfg_parameter_net["input_dim"]]
        inputs_x = inputs[:, self.cfg_parameter_net["input_dim"]:]

        parameter_net_output = self.parameter_net(inputs_parameter, training=training)
        split_idx = self.cfg_shape_net["input_dim"] * self.cfg_shape_net["units"] + self.cfg_shape_net["l_sx"] * self.cfg_shape_net["units"]**2 + self.cfg_shape_net["output_dim"] * self.cfg_shape_net["units"]
        weights = parameter_net_output[:, :split_idx]
        biases = parameter_net_output[:, split_idx:]
        shape_net_output = self.shape_net([inputs_x, weights, biases])
        
        return shape_net_output[0]