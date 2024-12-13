import tensorflow as tf
import numpy as np
import time
import logging
import contextlib
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import os

import nif.tf as nif
from nif.tf.optimizers import centralized_gradients_for_optimizer

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
enable_multi_gpu = False
enable_mixed_precision = False
nepoch = 5000
lr = 5e-3
batch_size = 512
checkpt_epoch = 1000
display_epoch = 100
print_figure_epoch = 100

from nif.tf.demo import TravelingWave
tw = TravelingWave()
train_data = tw.data

num_total_data = train_data.shape[0]
train_dataset = tf.data.Dataset.from_tensor_slices((train_data[:, :2], train_data[:, -1:]))
train_dataset = train_dataset.shuffle(num_total_data).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

if enable_mixed_precision:
    mixed_policy = "mixed_float16"
    # we might need this for `model.fit` to automatically do loss scaling
    policy = nif.mixed_precision.Policy(mixed_policy)
    nif.mixed_precision.set_global_policy(policy)
else:
    mixed_policy = 'float32'

class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_begin_time = time.time()
        self.history_loss = []
        logging.basicConfig(filename='./log', level=logging.INFO, format='%(message)s')

    def on_epoch_begin(self, epoch, logs=None):
        self.ts = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % display_epoch == 0:
            tnow = time.time()
            te = tnow - self.ts
            logging.info("Epoch {:6d}: avg.loss pe = {:4.3e}, {:d} points/sec, time elapsed = {:4.3f} hours".format(
                epoch, logs['loss'], int(batch_size / te), (tnow - self.train_begin_time) / 3600.0))
            self.history_loss.append(logs['loss'])
        if epoch % print_figure_epoch == 0:
            plt.figure()
            plt.semilogy(self.history_loss)
            plt.xlabel('epoch: per {} epochs'.format(print_figure_epoch))
            plt.ylabel('MSE loss')
            plt.savefig('./loss.png')
            plt.close()

            u_pred = self.model.predict(train_data[:,0:2]).reshape(10,200)
            fig,axs=plt.subplots(1,3,figsize=(16,4))
            im1=axs[0].contourf(tt, xx, train_data[:,-1].reshape(10,200),vmin=-5,vmax=5,levels=50,cmap='seismic')
            plt.colorbar(im1,ax=axs[0])

            im2=axs[1].contourf(tt, xx, u_pred,vmin=-5,vmax=5,levels=50,cmap='seismic')
            plt.colorbar(im2,ax=axs[1])

            im3=axs[2].contourf(tt, xx, (u_pred-train_data[:,-1].reshape(10,200)),vmin=-5,vmax=5,levels=50,cmap='seismic')
            plt.colorbar(im3,ax=axs[2])

            axs[0].set_xlabel('t')
            axs[0].set_ylabel('x')
            axs[0].set_title('true')
            axs[1].set_title('pred')
            axs[2].set_title('error')
            plt.savefig('vis.png')
            plt.close()

        if epoch % checkpt_epoch == 0 or epoch == nepoch - 1:
            print('save checkpoint epoch: %d...' % epoch)
            self.model.save_weights("./saved_weights/ckpt-{}.weights.h5".format(epoch))

def scheduler(epoch, lr):
    if epoch < 1000:
        return lr
    elif epoch < 2000:
        return 1e-3
    elif epoch < 4000:
        return 5e-4
    else:
        return 1e-4
        
scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

cm = tf.distribute.MirroredStrategy().scope() if enable_multi_gpu else contextlib.nullcontext()
with cm:
    optimizer = nif.optimizers.AdaBeliefOptimizer(lr)
    optimizer.get_gradients = centralized_gradients_for_optimizer(optimizer)

    model_ori = nif.NIF(cfg_shape_net, cfg_parameter_net, mixed_policy)
    model_opt = model_ori.build()

    model_opt.compile(optimizer, loss='mse')

# Create directory for saved weights if it doesn't exist
os.makedirs('./saved_weights', exist_ok=True)

# callbacks = []
callbacks = [LossAndErrorPrintingCallback(), scheduler_callback]
# callbacks = [tensorboard_callback, ]
# callbacks = [tensorboard_callback, LossAndErrorPrintingCallback(), scheduler_callback]
model_opt.fit(train_dataset, epochs=nepoch, batch_size=batch_size,
          shuffle=False, verbose=0, callbacks=callbacks)