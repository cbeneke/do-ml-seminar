import numpy as np
import contextlib
from matplotlib import pyplot as plt
import os

USE_TENSORFLOW = False

if USE_TENSORFLOW:
    import tensorflow as tf
    import nif.tf as nif
    from nif.tf.optimizers import centralized_gradients_for_optimizer
else:
    import torch
    import nif.torch as nif
    from nif.torch.optimizers import centralized_gradients_for_optimizer

enable_multi_gpu = False
enable_mixed_precision = False
nepoch = 5000
lr = 5e-3
batch_size = 512

NT=10 # 20
NX=200

x = np.linspace(0,1,NX,endpoint=False)
t = np.linspace(0,100,NT,endpoint=False)

xx,tt=np.meshgrid(x,t)

omega = 4
c = 0.12/20
x0 = 0.2

u = np.exp(-1000*(xx-x0-c*tt)**2)*np.sin(omega*(xx-x0-c*tt))

# vis
plt.figure()
for i in range(NT):
    plt.plot(x,u[i,:],'-',label=str(i) + '-th time')

plt.xlabel('$x$',fontsize=25)
plt.ylabel('$u$',fontsize=25)

# vis iso
plt.figure(figsize=(4,4))
ax = plt.axes(projection='3d')
ax.plot_surface(xx,tt,u,cmap="rainbow", lw=2)#,rstride=1, cstride=1)
ax.view_init(57, -80)
ax.set_xlabel(r'$x$',fontsize=25)
ax.set_ylabel(r'$t$',fontsize=25)
ax.set_zlabel(r'$u$',fontsize=25)

plt.tight_layout()

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

def scheduler(epoch, lr):
    if epoch < 1000:
        return lr
    elif epoch < 2000:
        return 1e-3
    elif epoch < 4000:
        return 5e-4
    else:
        return 1e-4

from nif.data import TravelingWave
tw = TravelingWave()
train_data = tw.data

num_total_data = train_data.shape[0]


## TensorFlow
if USE_TENSORFLOW:
    train_inputs = train_data[:, :2]
    train_targets = train_data[:, -1:]
    train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets))
    train_dataset = train_dataset.shuffle(num_total_data).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    cm = tf.distribute.MirroredStrategy().scope() if enable_multi_gpu else contextlib.nullcontext()
    with cm:
        optimizer = nif.optimizers.AdaBeliefOptimizer(lr)
        optimizer.get_gradients = centralized_gradients_for_optimizer(optimizer)

        model_ori = nif.NIF(cfg_shape_net, cfg_parameter_net, mixed_policy)
        model_opt = model_ori.build()

        model_opt.compile(optimizer, loss='mse')

    # Create directory for saved weights if it doesn't exist
    os.makedirs('./saved_weights', exist_ok=True)

    # Initialize callbacks
    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    loss_callback = nif.utils.LossAndErrorPrintingCallback(nepoch, train_data, xx, tt)
    callbacks = [loss_callback, scheduler_callback]

    # Train model
    model_opt.fit(train_dataset, epochs=nepoch, batch_size=batch_size,
            shuffle=False, verbose=0, callbacks=callbacks)
## PyTorch
else:
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