import time
import logging
import tensorflow as tf

display_epoch = 100
print_figure_epoch = 100
batch_size = 512

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