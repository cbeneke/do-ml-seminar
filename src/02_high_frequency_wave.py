import os
import tensorflow as tf
from nif import base
from nif import basetf

NIF_IMPLEMENTATION = os.getenv("NIF_IMPLEMENTATION", "functional")
OPTIMIZER = os.getenv("OPTIMIZER", "Adam")

enable_mixed_precision, nepoch, lr, batch_size, display_epoch, print_figure_epoch, NT, NX = base.get_base_configs()
u, x, t, x0, c, omega, xx, tt = base.setup_example_base(NT, NX)
dudx_1d, dudt_1d = base.get_derivative_data(x0, c, omega, xx, tt)

######
# Setup for NIF with SIREN
if NIF_IMPLEMENTATION == "upstream":
    import nif.upstream as nif
elif NIF_IMPLEMENTATION == "functional":
    import nif.functional as nif
else:
    raise ValueError(f"Invalid NIF implementation: {NIF_IMPLEMENTATION}")

cfg_shape_net = {
    "use_resblock": False,
    "connectivity": 'full',
    "input_dim": 1,
    "output_dim": 1,
    "units": 30,
    "nlayers": 2,
    "weight_init_factor": 0.01,
    "omega_0": 30.0,
    "activation": 'sine'
}
cfg_parameter_net = {
    "use_resblock": False,
    "input_dim": 1,
    "latent_dim": 1,
    "units": 30,
    "nlayers": 2,
    "activation": 'swish'
}

if enable_mixed_precision:
    mixed_policy = "mixed_float16"
    # we might need this for `model.fit` to automatically do loss scaling
    policy = tf.keras.mixed_precision.Policy(mixed_policy)
    tf.keras.mixed_precision.set_global_policy(policy)
else:
    mixed_policy = 'float32'
    
coef_grad = 1e-3

from nif.data import TravelingWaveHighFreq
tw = TravelingWaveHighFreq()

train_data = tw.data

num_total_data = train_data.shape[0]

train_inputs = train_data[:, :2]
train_targets = train_data[:, -1:]
train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets))
train_dataset = train_dataset.shuffle(num_total_data).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

if OPTIMIZER == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
elif OPTIMIZER == "adabelief":
    from nif.upstream.optimizers import AdaBeliefOptimizer, centralized_gradients_for_optimizer
    optimizer = AdaBeliefOptimizer(lr)
    optimizer.get_gradients = centralized_gradients_for_optimizer(optimizer)
else:
    raise ValueError(f"Invalid optimizer: {OPTIMIZER}")

model = nif.NIF(cfg_shape_net, cfg_parameter_net, mixed_policy)
model.build(input_shape=(cfg_shape_net["input_dim"] + cfg_parameter_net["input_dim"],))
model.compile(optimizer, loss='mse')

model.summary()
    
# Create directory for saved weights if it doesn't exist
os.makedirs('./saved_weights', exist_ok=True)

# Initialize callbacks
scheduler_callback = tf.keras.callbacks.LearningRateScheduler(base.scheduler)
loss_callback = basetf.LossAndErrorPrintingCallback(nepoch, train_data, xx, tt, NT, NX)
callbacks = [loss_callback, scheduler_callback]

model.fit(train_dataset,  verbose=0,epochs=nepoch, callbacks=callbacks)