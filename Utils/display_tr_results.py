import matplotlib.pyplot as plt
import numpy as np


def plot_me(dsrc_tr, dsrc_tt, save_path):
    
    data_tr = np.loadtxt(dsrc_tr, delimiter=',')
    data_tt = np.loadtxt(dsrc_tt, delimiter=',')
    plt.figure( figsize=(10, 10) )
    a = 0; b = len(data_tr[:, 0]) #- 1
    #plt.subplot(2,2,1) 
    plt.plot(range(a, b), data_tr[a:b, 1], color='blue', label='class train'); plt.legend()
    plt.plot(range(a, b), data_tt[a:b, 1], color='red',label='class  val'); plt.legend()
    
    #plt.xticks([])
    #plt.yticks([])
    #plt.axis('on')

    plt.savefig(save_path)
    plt.show()


