import contextlib
import os
import tensorflow as tf
from nif.upstream.optimizers import AdaBeliefOptimizer, centralized_gradients_for_optimizer
from nif import base

#NIF_IMPLEMENTATION="upstream"
NIF_IMPLEMENTATION="functional"

enable_multi_gpu, enable_mixed_precision, nepoch, lr, batch_size, checkpt_epoch, display_epoch, print_figure_epoch, NT, NX = base.get_base_configs()
u, x, t, x0, c, omega, xx, tt = base.setup_example_base(NT, NX)
dudx_1d, dudt_1d = base.get_derivative_data(x0, c, omega, xx, tt)

######
# Setup for basic NIF
if NIF_IMPLEMENTATION == "upstream":
    import nif.upstream as nif
elif NIF_IMPLEMENTATION == "functional":
    import nif.functional as nif

cfg_shape_net = {
    "connectivity": 'full',
    "input_dim": 1,
    "output_dim": 1,
    "units": 30,
    "nlayers": 2,
    "activation": 'swish'
}
cfg_parameter_net = {
    "input_dim": 1,
    "latent_dim": 1,
    "units": 30,
    "nlayers": 2,
    "activation": 'swish',
}

if enable_mixed_precision:
    mixed_policy = "mixed_float16"
    # we might need this for `model.fit` to automatically do loss scaling
    policy = nif.mixed_precision.Policy(mixed_policy)
    nif.mixed_precision.set_global_policy(policy)
else:
    mixed_policy = 'float32'

from nif.data import TravelingWave
tw = TravelingWave()
train_data = tw.data

num_total_data = train_data.shape[0]


train_inputs = train_data[:, :2]
train_targets = train_data[:, -1:]
train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets))
train_dataset = train_dataset.shuffle(num_total_data).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

cm = tf.distribute.MirroredStrategy().scope() if enable_multi_gpu else contextlib.nullcontext()
with cm:
    optimizer = AdaBeliefOptimizer(lr)
    optimizer.get_gradients = centralized_gradients_for_optimizer(optimizer)

    model = nif.NIF(cfg_shape_net, cfg_parameter_net, mixed_policy)
    model.build(input_shape=(cfg_shape_net["input_dim"] + cfg_parameter_net["input_dim"],))
    model.compile(optimizer, loss='mse')

    model.summary()

# Create directory for saved weights if it doesn't exist
os.makedirs('./saved_weights', exist_ok=True)

# Initialize callbacks
scheduler_callback = tf.keras.callbacks.LearningRateScheduler(base.scheduler)
loss_callback = base.LossAndErrorPrintingCallback(nepoch, train_data, xx, tt, NT, NX)
callbacks = [loss_callback, scheduler_callback]

# Train model
model.fit(train_dataset, epochs=nepoch, batch_size=batch_size,
        shuffle=False, verbose=0, callbacks=callbacks)
