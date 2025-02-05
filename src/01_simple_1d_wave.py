import contextlib
import os
from nif import base

NIF_IMPLEMENTATION = os.getenv("NIF_IMPLEMENTATION", "functional")

if NIF_IMPLEMENTATION == "upstream" or NIF_IMPLEMENTATION == "functional":
    import tensorflow as tf
    from nif.upstream.optimizers import AdaBeliefOptimizer, centralized_gradients_for_optimizer
    from nif import basetf
elif NIF_IMPLEMENTATION == "pytorch":
    import torch
    from nif.torch import utils
else:
    raise ValueError(f"Invalid NIF implementation: {NIF_IMPLEMENTATION}")

enable_multi_gpu, enable_mixed_precision, nepoch, lr, batch_size, display_epoch, print_figure_epoch, NT, NX = base.get_base_configs()
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
    "activation": 'swish',
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

if NIF_IMPLEMENTATION == "upstream":
    import nif.upstream as nif
elif NIF_IMPLEMENTATION == "functional":
    import nif.functional as nif
elif NIF_IMPLEMENTATION == "pytorch":
    import nif.torch as nif
else:
    raise ValueError(f"Invalid NIF implementation: {NIF_IMPLEMENTATION}")
    

from nif.data import TravelingWave
tw = TravelingWave()
train_data = tw.data

if NIF_IMPLEMENTATION == "upstream" or NIF_IMPLEMENTATION == "functional":
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
    loss_callback = basetf.LossAndErrorPrintingCallback(nepoch, train_data, xx, tt, NT, NX)
    callbacks = [loss_callback, scheduler_callback]

    # Train model
    model.fit(train_dataset, epochs=nepoch, batch_size=batch_size,
            shuffle=False, verbose=0, callbacks=callbacks)
elif NIF_IMPLEMENTATION == "pytorch":
    train_inputs = torch.from_numpy(train_data[:, :2]).float()
    train_targets = torch.from_numpy(train_data[:, -1:]).float()
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)

    # Initialize model, optimizer, and data loader
    model = nif.NIF(cfg_shape_net, cfg_parameter_net, mixed_policy)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # Create logger
    logger = nif.TrainingLogger(
        display_epoch=100,
        print_figure_epoch=100,
        checkpt_epoch=1000,
        n_epochs=nepoch
    )

    # Train model
    history = nif.train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        n_epochs=nepoch,
        logger=logger
    )
else:
    raise ValueError(f"Invalid NIF implementation: {NIF_IMPLEMENTATION}")
