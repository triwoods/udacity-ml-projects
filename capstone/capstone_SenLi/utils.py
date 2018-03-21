import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, convolve

def q15_to_lsf(x):
    return x.astype('float') * np.pi / np.power(2, 15)


def lsf2poly(L):
    order = len(L)
    Q = L[::2]
    P = L[1::2]
    poles_P = np.r_[np.exp(1j*P),np.exp(-1j*P)]
    poles_Q = np.r_[np.exp(1j*Q),np.exp(-1j*Q)]
    
    P = np.poly(poles_P)
    Q = np.poly(poles_Q)
    
    P = convolve(P, np.array([1.0, -1.0]))
    Q = convolve(Q, np.array([1.0, 1.0]))
    
    a = 0.5*(P+Q)
    return a[:-1]

    
def plot_lsf(lsfs):
    num_plots = lsfs.shape[0]
    plt.figure(figsize=(7, 7))
    for i in range(num_plots):
        x_mark = lsfs[i, :]
        y_mark = np.zeros_like(x_mark)
        a = lsf2poly(x_mark)
        w, h = freqz(1.0, a)
        plt.subplot(num_plots, 1, i+1)
        plt.plot(w, 20 * np.log10(abs(h)))
        plt.plot(x_mark, y_mark, 'x')
        plt.axis([0, np.pi/2, -10, 100])
        # plt.tight_layout()
    return


def plot_lsf_hist(lsfs):
    num_dims = lsfs.shape[1]
    plt.figure(figsize=(7, 7))
    for i in range(num_dims):
        plt.subplot(num_dims, 1, i+1)
        plt.hist(lsfs[:, i], bins=50, normed=True)
    return