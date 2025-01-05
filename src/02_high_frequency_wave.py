import numpy as np
import contextlib
import os
import tensorflow as tf
from nif import base

NIF_IMPLEMENTATION="upstream"
#NIF_IMPLEMENTATION="functional"

enable_multi_gpu, enable_mixed_precision, nepoch, lr, batch_size, checkpt_epoch, display_epoch, print_figure_epoch, NT, NX = base.get_base_configs()
u, x, t, x0, c, omega, xx, tt = base.setup_example_base(NT, NX)
dudx_1d, dudt_1d = base.get_derivative_data(x0, c, omega, xx, tt)

######
# Setup for NIF with SIREN
if NIF_IMPLEMENTATION == "upstream":
    import nif.upstream as nif
    from nif.upstream.layers import JacobianLayer
elif NIF_IMPLEMENTATION == "functional":
    import nif.functional as nif

cfg_shape_net = {
    "use_resblock":False,
    "connectivity": 'full',
    "input_dim": 1,
    "output_dim": 1,
    "units": 30,
    "nlayers": 2,
    "weight_init_factor": 0.01,
    "omega_0":30.0
}
cfg_parameter_net = {
    "use_resblock":False,
    "input_dim": 1,
    "latent_dim": 1,
    "units": 30,
    "nlayers": 2,
    "activation": 'swish'
}

if enable_mixed_precision:
    mixed_policy = "mixed_float16"
    # we might need this for `model.fit` to automatically do loss scaling
    policy = nif.mixed_precision.Policy(mixed_policy)
    nif.mixed_precision.set_global_policy(policy)
else:
    mixed_policy = 'float32'
    
coef_grad = 1e-3

from nif.data import TravelingWaveHighFreq
tw = TravelingWaveHighFreq()

# no augment 
train_data_ng = tw.data
train_mean_ng = tw.mean
train_std_ng = tw.std

train_data = np.hstack([tw.data, dudt_1d/(tw.std[2]/tw.std[0]), dudx_1d/(tw.std[2]/tw.std[1])])

train_mean = np.hstack([tw.mean,0,0])
train_std = np.hstack([tw.std, tw.std[2]/tw.std[0], tw.std[2]/tw.std[1]])

num_total_data = train_data.shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices((train_data[:, :2], train_data[:, 2:5]))
train_dataset = train_dataset.shuffle(num_total_data).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

cm = tf.distribute.MirroredStrategy().scope() if enable_multi_gpu else contextlib.nullcontext()
with cm:
    optimizer = tf.keras.optimizers.Adam(lr)
    model = nif.NIFMultiScale(cfg_shape_net, cfg_parameter_net, mixed_policy)
    model.build(input_shape=None)
    
    # sobolov training
    x_index = [0, 1]  # t,x
    y_index = [0]  # we have 1 field output - u

    n_output = 1

    y_and_dydx = JacobianLayer(model, y_index, x_index)
    y, dy_dx = y_and_dydx(model.inputs[0])  ##  use[0] to make sure shape is good
    dy_dx_1d = tf.reshape(dy_dx, [-1,2*n_output])
    y_and_dydx_1d = tf.concat([y, dy_dx_1d],-1)
    model = tf.keras.Model([model.inputs[0]], [y_and_dydx_1d])

    class Sobolov_MSE(tf.keras.losses.Loss):
        def call(self, y_true, y_pred):
            sd_field = tf.square(y_true[:, :n_output] - y_pred[:, :n_output])
            # sd_grad = tf.square(y_true[:, n_output:] - y_pred[:, n_output:])
            sd_grad =  tf.square(y_true[:, n_output+1:] - y_pred[:, n_output+1:])
            return tf.reduce_mean(sd_field,axis=-1) + coef_grad*tf.reduce_mean(sd_grad,axis=-1)
    
    def Sobolov_MSE_u(y_true, y_pred):
            sd_field = tf.square(y_true[:, :n_output] - y_pred[:, :n_output])
            return tf.reduce_mean(sd_field,axis=-1)

    def Sobolov_MSE_dudt(y_true, y_pred):
        sd_grad = tf.square(y_true[:, n_output:n_output+1] - y_pred[:, n_output:n_output+1])
        return coef_grad*tf.reduce_mean(sd_grad,axis=-1)

    def Sobolov_MSE_dudx(y_true, y_pred):
        sd_grad = tf.square(y_true[:, n_output+1:] - y_pred[:, n_output+1:])
        return coef_grad*tf.reduce_mean(sd_grad,axis=-1)
    
    model.compile(optimizer, loss=Sobolov_MSE(), 
                      metrics=[Sobolov_MSE_u, Sobolov_MSE_dudt, Sobolov_MSE_dudx])
    
# Create directory for saved weights if it doesn't exist
os.makedirs('./saved_weights', exist_ok=True)

# Initialize callbacks
scheduler_callback = tf.keras.callbacks.LearningRateScheduler(base.scheduler)
loss_callback = base.LossAndErrorPrintingCallback(nepoch, train_data, xx, tt, NT, NX)
callbacks = [loss_callback, scheduler_callback]

model.fit(train_dataset,  verbose=0,epochs=nepoch, callbacks=callbacks)