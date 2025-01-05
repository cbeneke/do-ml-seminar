import tensorflow as tf

from nif.tf_v2.layers.shortcut import Shortcut
from nif.tf_v2.layers.resnet import ResNet
from nif.tf_v2.layers.static_dense import StaticDense
from nif.tf_v2 import utils

class NIF(tf.keras.Model):
    def __init__(self, cfg_shape_net, cfg_parameter_net, mixed_policy):
        self.mixed_policy = tf.keras.mixed_precision.Policy(
            mixed_policy
        )  # policy object can be feed into keras.layer
        self.variable_Dtype = self.mixed_policy.variable_dtype
        self.compute_Dtype = self.mixed_policy.compute_dtype

        super().__init__(dtype=self.mixed_policy)

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
            self.mixed_policy,
            self.pnet_kernel_regularizer,
            self.pnet_bias_regularizer,
            self.pnet_act_regularizer,
        )

        # Build shape net
        self.shape_net_layers = self._build_shape_net(self.cfg_shape_net)

    @staticmethod
    def _build_parameter_net(cfg_parameter_net, cfg_shape_net, mixed_policy, kernel_regularizer, bias_regularizer, activity_regularizer) -> list:
        layers = []

        # First Layer
        layers.append(tf.keras.layers.Dense(
            name="first_pnet",
            units=cfg_parameter_net["units"],
            activation=tf.keras.activations.get(cfg_parameter_net["activation"]),
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dtype=mixed_policy,
        ))

        # Hidden Layers - ResNet or Shortcut based on config
        HiddenLayer = ResNet if cfg_parameter_net.get("use_resblock", False) else Shortcut
        for i in range(cfg_parameter_net["nlayers"]):
            layers.append(HiddenLayer(
                name=f"hidden_pnet_{i}",
                units=cfg_parameter_net["units"],
                activation=cfg_parameter_net["activation"],
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                dtype=mixed_policy,
            ))
        
        # Bottleneck layer
        layers.append(tf.keras.layers.Dense(
            name="bottleneck_pnet",
            units=cfg_parameter_net["latent_dim"],
            activation=cfg_parameter_net["activation"],
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dtype=mixed_policy,
        ))

        # Last Layer with additional activation regularizer
        layers.append(tf.keras.layers.Dense(
            name="last_pnet",
            units=utils.get_parameter_net_output_dim(cfg_shape_net),
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            dtype=mixed_policy,
        ))
        return layers

    @staticmethod
    def _build_shape_net(cfg_shape_net) -> list:
        layers = []

        # First layer -- input_dim -> units fully connected
        layers.append(StaticDense(
            name="first_snet",
            units=cfg_shape_net["units"],
            cfg_shape_net=cfg_shape_net,
            weights_from=0,
            weights_to=cfg_shape_net["input_dim"] * cfg_shape_net["units"],
            biases_from=0,
            biases_to=cfg_shape_net["units"],
        ))

        for i in range(cfg_shape_net["nlayers"]):
            # Hidden layer -- units -> units fully connected
            layers.append(StaticDense(
                name=f"hidden_snet_{i}",
                units=cfg_shape_net["units"],
                cfg_shape_net=cfg_shape_net,
                weights_from=cfg_shape_net["input_dim"] * cfg_shape_net["units"] + i * cfg_shape_net["units"]**2,
                weights_to=cfg_shape_net["input_dim"] * cfg_shape_net["units"] + (i + 1) * cfg_shape_net["units"]**2,
                biases_from=(i+1) * cfg_shape_net["units"],
                biases_to=(i+2) * cfg_shape_net["units"],
            ))
        # Last layer -- units -> output_dim fully connected
        layers.append(StaticDense(
            name="last_snet",
            units=cfg_shape_net["output_dim"],
            cfg_shape_net=cfg_shape_net,
            weights_from=cfg_shape_net["input_dim"] * cfg_shape_net["units"] + cfg_shape_net["nlayers"] * cfg_shape_net["units"]**2,
            weights_to=cfg_shape_net["input_dim"] * cfg_shape_net["units"] + cfg_shape_net["nlayers"] * cfg_shape_net["units"]**2 + cfg_shape_net["output_dim"] * cfg_shape_net["units"],
            biases_from=cfg_shape_net["nlayers"] * cfg_shape_net["units"],
            biases_to=cfg_shape_net["nlayers"] * cfg_shape_net["units"] + cfg_shape_net["output_dim"],
        ))
        return layers

    def build(self, input_shape):
        super().build(input_shape)

        # Build parameter net layers
        x_shape = (None, self.cfg_parameter_net["input_dim"])
        for layer in self.parameter_net_layers:
            layer.build(x_shape)
            x_shape = layer.compute_output_shape(x_shape)

        # Build shape net layers
        x_shape = (None, self.cfg_shape_net["input_dim"])
        for layer in self.shape_net_layers:
            layer.build(x_shape)
            x_shape = layer.compute_output_shape(x_shape)

        self.built = True

    def call(self, inputs, training=None, mask=None):
        # Call parameter net
        parameter_net_output = inputs[:, :self.cfg_parameter_net["input_dim"]]
        for layer in self.parameter_net_layers:
            parameter_net_output = layer(parameter_net_output, training=training)

        # Call shape net
        shape_net_output = inputs[:, self.cfg_parameter_net["input_dim"]:]
        for layer in self.shape_net_layers:
            layer.pass_parameters(parameter_net_output)
            shape_net_output = layer(shape_net_output, training=False)

        return shape_net_output
    