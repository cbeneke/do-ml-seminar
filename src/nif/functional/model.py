import tensorflow as tf

from nif.functional.layers.shortcut import Shortcut
from nif.functional.layers.resnet import ResNet
from nif.functional.layers.hyper_dense import HyperDense
from nif.functional.layers.hyper_siren import HyperSIREN
from nif.functional import utils

class SIRENKernelInitializer(tf.keras.initializers.Initializer):
    def __init__(self, cfg_shape_net):
        self.cfg_shape_net = cfg_shape_net
        self.input_dim = cfg_shape_net["input_dim"]
        self.output_dim = cfg_shape_net["output_dim"]
        self.units = cfg_shape_net["units"]
        self.layers = cfg_shape_net["nlayers"]
        self.omega_0 = cfg_shape_net["omega_0"]

    def __call__(self, shape, dtype=None):
        weights = tf.zeros(
            shape=(shape[0], utils.get_weights_dim(self.cfg_shape_net), shape[2]),
            dtype=dtype,
        )

        # First Layer - uniform distribution between +- 1/input_dim
        first_layer_cut = self.input_dim * self.units
        weights[:, :first_layer_cut ] = tf.random.uniform(
            shape=(shape[0], first_layer_cut, shape[2]),
            minval=-1,
            maxval=1,
            dtype=dtype,
        )

        # Hidden Layers - uniform distribution between +- sqrt(6/input_dim) * omega_0
        for i in tf.range(self.layers):
            hidden_layer_from = first_layer_cut + (i * (self.units**2))
            hidden_layer_to = first_layer_cut + ((i+1) * (self.units**2))
            weights[:, hidden_layer_from:hidden_layer_to ] = tf.random.uniform(
                shape=(shape[0], self.units**2, shape[2]),
                minval=-self.omega_0 * tf.sqrt(6 / self.input_dim),
                maxval=self.omega_0 * tf.sqrt(6 / self.input_dim),
                dtype=dtype,
            )

        # Last Layer
        last_layer_from = first_layer_cut + (self.layers * (self.units**2))
        weights[:, last_layer_from: ] = tf.random.uniform(
            shape=(shape[0], self.output_dim * self.units, shape[2]),
            minval=-self.omega_0 * tf.sqrt(6 / self.input_dim),
            maxval=self.omega_0 * tf.sqrt(6 / self.input_dim),
            dtype=dtype,
        )

        return weights

class NIF(tf.keras.Model):
    def __init__(self, cfg_shape_net, cfg_parameter_net, mixed_policy):
        super().__init__()

        # Store the config
        self.cfg_shape_net = cfg_shape_net
        self.cfg_parameter_net = cfg_parameter_net

        # Build parameter net
        self.parameter_net_layers = self._build_parameter_net(
            self.cfg_parameter_net,
            self.cfg_shape_net,
            mixed_policy,
        )

        # Build shape net
        self.shape_net_layers = self._build_shape_net(self.cfg_shape_net)

    @staticmethod
    def _build_parameter_net(cfg_parameter_net, cfg_shape_net, mixed_policy) -> list:

        # Setup for standard regularization
        # 1. regularization for kernel in parameter net
        if isinstance(cfg_parameter_net.get("l2_reg", None), (float, int)):
            kernel_regularizer = tf.keras.regularizers.L2(cfg_parameter_net["l2_reg"])
            bias_regularizer = tf.keras.regularizers.L2(cfg_parameter_net["l2_reg"])
        elif isinstance(cfg_parameter_net.get("l1_reg", None), (float, int)):
            kernel_regularizer = tf.keras.regularizers.L1(cfg_parameter_net["l1_reg"])
            bias_regularizer = tf.keras.regularizers.L1(cfg_parameter_net["l1_reg"])
        else:
            kernel_regularizer = None
            bias_regularizer = None

        # 2. output of parameter net regularization
        if isinstance(cfg_parameter_net.get("act_l2_reg", None), (float, int)):
            activity_regularizer = tf.keras.regularizers.L2(cfg_parameter_net["act_l2_reg"])
        elif isinstance(cfg_parameter_net.get("act_l1_reg", None), (float, int)):
            activity_regularizer = tf.keras.regularizers.L1(cfg_parameter_net["act_l1_reg"])
        else:
            activity_regularizer = None

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
        
        # # Bottleneck layer
        # layers.append(tf.keras.layers.Dense(
        #     name="bottleneck_pnet",
        #     units=cfg_parameter_net["latent_dim"],
        #     activation=cfg_parameter_net["activation"],
        #     kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
        #     bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
        #     kernel_regularizer=kernel_regularizer,
        #     bias_regularizer=bias_regularizer,
        #     dtype=mixed_policy,
        # ))

        # Last Layer with additional activation regularizer
        if cfg_parameter_net["activation"] == 'sine':
            last_layer_kernel_initializer = SIRENKernelInitializer(cfg_shape_net)
        else:
            last_layer_kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.1)

        layers.append(tf.keras.layers.Dense(
            name="last_pnet",
            units=utils.get_parameter_net_output_dim(cfg_shape_net),
            kernel_initializer=last_layer_kernel_initializer,
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
        if cfg_shape_net["activation"] == 'sine':
            ShapeNetLayer = HyperSIREN
            extra_args = {"omega_0": cfg_shape_net["omega_0"]}
        else:
            ShapeNetLayer = HyperDense
            extra_args = {}

        # First layer -- input_dim -> units fully connected
        layers.append(ShapeNetLayer(
            name="first_snet",
            units=cfg_shape_net["units"],
            activation=cfg_shape_net["activation"],
            **extra_args,
            weights_from=0,
            weights_to=cfg_shape_net["input_dim"] * cfg_shape_net["units"],
            bias_offset=utils.get_weights_dim(cfg_shape_net),
            biases_from=0,
            biases_to=cfg_shape_net["units"],
        ))

        for i in range(cfg_shape_net["nlayers"]):
            # Hidden layer -- units -> units fully connected
            layers.append(ShapeNetLayer(
                name=f"hidden_snet_{i}",
                units=cfg_shape_net["units"],
                activation=cfg_shape_net["activation"],
                **extra_args,
                weights_from=cfg_shape_net["input_dim"] * cfg_shape_net["units"] + i * cfg_shape_net["units"]**2,
                weights_to=cfg_shape_net["input_dim"] * cfg_shape_net["units"] + (i + 1) * cfg_shape_net["units"]**2,
                bias_offset=utils.get_weights_dim(cfg_shape_net),
                biases_from=(i+1) * cfg_shape_net["units"],
                biases_to=(i+2) * cfg_shape_net["units"],
            ))
            
        # Last layer -- units -> output_dim fully connected
        layers.append(ShapeNetLayer(
            name="last_snet",
            units=cfg_shape_net["output_dim"],
            activation=None,
            **extra_args,
            weights_from=cfg_shape_net["input_dim"] * cfg_shape_net["units"] + cfg_shape_net["nlayers"] * cfg_shape_net["units"]**2,
            weights_to=cfg_shape_net["input_dim"] * cfg_shape_net["units"] + cfg_shape_net["nlayers"] * cfg_shape_net["units"]**2 + cfg_shape_net["output_dim"] * cfg_shape_net["units"],
            bias_offset=utils.get_weights_dim(cfg_shape_net),
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

    @tf.function
    def call(self, inputs, training=None, mask=None):
        # Call parameter net
        parameter_net_output = inputs[:, :self.cfg_parameter_net["input_dim"]]
        for layer in self.parameter_net_layers:
            parameter_net_output = layer(parameter_net_output, training=training)

        # Call shape net
        shape_net_output = inputs[:, self.cfg_parameter_net["input_dim"]:]
        for layer in self.shape_net_layers:
            shape_net_output = layer([shape_net_output, parameter_net_output], training=False)

        return shape_net_output
    