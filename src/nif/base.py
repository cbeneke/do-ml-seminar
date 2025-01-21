import time
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def setup_example_base(NT, NX):
    x = np.linspace(0,1,NX,endpoint=False)
    t = np.linspace(0,100,NT,endpoint=False)

    xx,tt=np.meshgrid(x,t)

    omega = 400
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

    return u, x, t, x0, c, omega, xx, tt

def get_derivative_data(x0, c, omega, xx, tt):
    dudx = np.exp(-1000*(xx-x0-c*tt)**2)*(-2000*(xx-x0-c*tt)*np.sin(omega*(xx-x0-c*tt)) + 
                                        omega*np.cos(omega*(xx-x0-c*tt)))

    dudt = np.exp(-1000*(xx-x0-c*tt)**2)*(2000*c*(xx-x0-c*tt)* np.sin(omega*(xx-x0-c*tt)) - 
                                        omega*c* np.cos(omega*(xx-x0-c*tt)))


    dudx_1d = dudx.reshape(-1,1)
    dudt_1d = dudt.reshape(-1,1)

    return dudx_1d, dudt_1d

def get_base_configs():
    enable_multi_gpu = False
    enable_mixed_precision = False
    nepoch = 5000
    lr = 1e-4
    batch_size = 512
    display_epoch = 100
    print_figure_epoch = 100

    NT=10 # 20
    NX=200
    
    return enable_multi_gpu, enable_mixed_precision, nepoch, lr, batch_size, display_epoch, print_figure_epoch, NT, NX

def scheduler(epoch, lr):
    return lr
    # if epoch < 1000:
    #     return lr
    # elif epoch < 2000:
    #     return 1e-3
    # elif epoch < 4000:
    #     return 5e-4
    # else:
    #     return 1e-4
    
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
        display_epoch = 100
        print_figure_epoch = 100
        batch_size = 512
        checkpt_epoch = 1000
        
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