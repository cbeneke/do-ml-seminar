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
    # return lr
    if epoch < 1000:
        return lr
    elif epoch < 2000:
        return 1e-3
    elif epoch < 4000:
        return 5e-4
    else:
        return 1e-4