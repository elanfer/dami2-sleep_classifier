import numpy as np
import scipy.signal as signal


# Butterworth Filter
def butter_bandpass_filter(data, highpass, fs, order=4):
    b, a = butter_bandpass(0, highpass, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highpass, fs, order=4):
    nyq = 0.5 * fs
    #       low = lowcut / nyq
    high = highpass / nyq
    b, a = signal.butter(order, high, btype='highpass')
    return b, a


def powerfunction(data, sampling, win="hanning"):
    winlength = len(data)

    # define weight window (e.g Hanning Window)
    if win == "hanning":
        weight_win = np.hanning(winlength)

    elif win == False:
        weight_win = np.random.randint(1, 1 + 1, winlength)

    # define power function window size and
    fft_win = int(np.floor(winlength / 2))
    # define fft frequency range
    fft_ticks = np.arange(1, fft_win + 1, 1) / np.floor(winlength / sampling)
    # do fft
    fft_result = np.fft.fft(data)
    # calculate powert function
    powerfun = (pow(abs(fft_result), 2) * weight_win)[0:fft_win]
    return fft_ticks, powerfun


def random_sampling_ts(data, seed=123):
    winlength = len(data)
    np.random.seed(seed)
    rands = np.random.randint(0, winlength, winlength)
    sampled_data = data[rands]
    return sampled_data


def random_powerfunction(data, sampling, win="hanning", iter=10, seed=123):
    periodo_lenght = int(np.floor(len(data) / 2))
    rand_periodogram = np.zeros(periodo_lenght)
    for i in range(1, iter):
        rand_data = random_sampling_ts(data, (2 * i + 3))
        powerfun = powerfunction(rand_data, sampling, win)
        rand_periodogram = rand_periodogram + powerfun
    return rand_periodogram / iter
