import tensorflow as tf
from tensorflow.keras import mixed_precision

from .__about__ import __version__
from nif.tf import data
from nif.tf import demo
from nif.tf import optimizers
from tensorflow.model import NIF
from tensorflow.model import NIFMultiScale
from tensorflow.model import NIFMultiScaleLastLayerParameterized

gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_physical_devices("GPU")
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

__all__ = [
    "data",
    "tf",
    "NIFMultiScale",
    "NIFMultiScaleLastLayerParameterized",
    "NIF",
    "mixed_precision",
    "optimizers",
    "demo",
]
