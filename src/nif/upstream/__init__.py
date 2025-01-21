import tensorflow as tf
from tensorflow.keras import mixed_precision

from .__about__ import __version__
from nif.upstream import optimizers
from nif.upstream.model import NIF
from nif.upstream.model import NIFMultiScale
from nif.upstream.model import NIFMultiScaleLastLayerParameterized

gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_physical_devices("GPU")
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

__all__ = [
    "NIFMultiScale",
    "NIFMultiScaleLastLayerParameterized",
    "NIF",
    "mixed_precision",
    "optimizers",
    "utils",
]
