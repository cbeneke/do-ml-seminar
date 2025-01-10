def get_weights_dim(cfg_shape_net):
    return (
        cfg_shape_net["input_dim"] * cfg_shape_net["units"]  # First layer
        + cfg_shape_net["nlayers"] * cfg_shape_net["units"]**2  # Hidden layers
        + cfg_shape_net["output_dim"] * cfg_shape_net["units"]  # Last layer
    )

def get_biases_dim(cfg_shape_net):
    return (
        cfg_shape_net["units"]  # First layer
        + cfg_shape_net["nlayers"] * cfg_shape_net["units"]  # Hidden layers
        + cfg_shape_net["output_dim"]  # Last layer
    )

def get_parameter_net_output_dim(cfg_shape_net):
    return get_weights_dim(cfg_shape_net) + get_biases_dim(cfg_shape_net)