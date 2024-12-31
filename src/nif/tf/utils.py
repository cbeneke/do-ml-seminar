import time
import logging
import tensorflow as tf

import matplotlib.pyplot as plt

display_epoch = 100
print_figure_epoch = 100
batch_size = 512
checkpt_epoch = 1000


class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    def __init__(self, nepoch, train_data, xx, tt, NT, NX):
        super().__init__()
        self.train_data = train_data
        self.xx = xx 
        self.tt = tt
        self.nepoch = nepoch
        self.NT = NT
        self.NX = NX

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

            u_pred = self.model.predict(self.train_data[:,0:2]).reshape(self.NT,self.NX)
            fig,axs=plt.subplots(1,3,figsize=(16,4))
            im1=axs[0].contourf(self.tt, self.xx, self.train_data[:,-1].reshape(self.NT,self.NX),vmin=-5,vmax=5,levels=50,cmap='seismic')
            plt.colorbar(im1,ax=axs[0])

            im2=axs[1].contourf(self.tt, self.xx, u_pred,vmin=-5,vmax=5,levels=50,cmap='seismic')
            plt.colorbar(im2,ax=axs[1])

            im3=axs[2].contourf(self.tt, self.xx, (u_pred-self.train_data[:,-1].reshape(self.NT,self.NX)),vmin=-5,vmax=5,levels=50,cmap='seismic')
            plt.colorbar(im3,ax=axs[2])

            axs[0].set_xlabel('t')
            axs[0].set_ylabel('x')
            axs[0].set_title('true')
            axs[1].set_title('pred')
            axs[2].set_title('error')
            plt.savefig('vis.png')
            plt.close()

        if epoch % checkpt_epoch == 0 or epoch == self.nepoch - 1:
            print('save checkpoint epoch: %d...' % epoch)
            self.model.save_weights("./saved_weights/ckpt-{}.weights.h5".format(epoch))