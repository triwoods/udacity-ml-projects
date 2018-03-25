import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, convolve

# Data conversion and transformation
def q15_to_lsf(x):
    return x.astype('float') * np.pi / np.power(2, 15)

def deltas(x, n=1):
    xb = np.roll(x, n, axis=0)
    xf = np.roll(x, -n, axis=0)
    return (xf - xb)/2

def serialized(x, step_left=1, step_right=1):
    dim_x = x.shape[-1]
    total_steps = step_left + step_right + 1
    
    x = np.roll(x, -step_right, axis=0)
    y = x
    for i in range(step_left + step_right):
        xr = np.roll(x, 1, axis=0)
        y = np.hstack((xr, y))
        x = xr
    return y.reshape((-1, total_steps, dim_x))

def train_valid_split(X, y, step_left=0, step_right=0, valid_size=0.1, shuffle=False, random_state=0):
    X = serialized(X, step_left, step_right)
    data_size = X.shape[0]
    valid_len = int(data_size * valid_size)
    x = np.arange(data_size)
    if shuffle is True:
        np.random.seed(random_state)
        np.random.shuffle(x)
    index_train, index_valid = x[valid_len:], x[:valid_len]
    return X[index_train, :, :], X[index_valid, :, :], y[index_train, :], y[index_valid, :]

# Time utilities
def second_to_hms(second):
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    return h, m, s

def frame_to_hms(nf, frame_rate=0.020):
    second = nf * frame_rate
    h, m, s = second_to_hms(second)
    # print "%d:%02d:%.3f" % (h, m, s)
    hms_str = "%d:%02d:%.3f" % (h, m, s)
    return hms_str

# LPC funtions and plot
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
    plt.figure(figsize=(7, 3.5 * num_plots))
    for i in range(num_plots):
        x_mark = lsfs[i, :]
        y_mark = np.zeros_like(x_mark)
        a = lsf2poly(x_mark)
        w, h = freqz(1.0, a)
        plt.subplot(num_plots, 1, i+1)
        plt.plot(w, 20 * np.log10(abs(h)))
        plt.plot(x_mark, y_mark, 'x')
        plt.xlabel('Frequency in radius')
        plt.ylabel('Log power magnitude in dB')
        plt.legend(['spectral shape','LSF position'])
        plt.axis([0, np.pi/2, -10, 100])

def plot_lsf_hist(lsfs):
    num_plots = lsfs.shape[1]
    plt.figure(figsize=(8, 2 * num_plots))
    for i in range(num_plots):
        plt.subplot(num_plots, 1, i+1)
        plt.hist(lsfs[:, i], bins=30, normed=False)
        plt.xlabel('Frequency in radius')
        plt.ylabel('LSF[' + str(i) + '] count')
        plt.axis([0, np.pi/2, 0, 50000])

def plot_model_result(result, ylim=(0.005, 0.010)):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    bar_width = 0.45
    colors = ['#A00000','#00A0A0','#00A000']

    for j, metric in result.items():
        metric_names = list(metric.keys())
        metric_values = list(metric.values())
        for i in np.arange(3):
            ax[i%3].bar(j+bar_width, metric_values[i], width = bar_width, color = colors[i])
            ax[i%3].set_xticks([0.45, 1.45, 2.45])
            ax[i%3].set_xticklabels(["1%", "10%", "100%"])
            ax[i%3].set_xlabel("Training Set Size")
            ax[i%3].set_xlim((-0.1, 3.0))
            ax[i%3].set_ylabel("MSE")
            ax[i%3].set_ylim(ylim)
            ax[i%3].set_title(metric_names[i])
