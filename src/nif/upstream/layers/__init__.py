from tensorflow.keras.layers import Dense

from nif.upstream.layers.gradient import HessianLayer
from nif.upstream.layers.gradient import JacobianLayer
from nif.upstream.layers.gradient import JacRegLatentLayer
from nif.upstream.layers.mlp import BiasAddLayer
from nif.upstream.layers.mlp import EinsumLayer
from nif.upstream.layers.mlp import MLP_ResNet
from nif.upstream.layers.mlp import MLP_SimpleShortCut
from nif.upstream.layers.regularization import ParameterOutputL1ActReg
from nif.upstream.layers.siren import HyperLinearForSIREN
from nif.upstream.layers.siren import SIREN
from nif.upstream.layers.siren import SIREN_ResNet

__all__ = [
    "SIREN",
    "SIREN_ResNet",
    "Dense",
    "HyperLinearForSIREN",
    "MLP_ResNet",
    "MLP_SimpleShortCut",
    "JacRegLatentLayer",
    "JacobianLayer",
    "HessianLayer",
    "ParameterOutputL1ActReg",
    "EinsumLayer",
    "BiasAddLayer",
]
