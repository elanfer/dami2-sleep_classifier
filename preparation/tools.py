import os

import csv
import numpy as np
import scipy.interpolate as ip
import scipy.signal as signal
import statistics as stat


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


def get_periodogram(data, s_rate, win="hanning"):
    '''
    function to calculate the periododogram of a TSD

    :param data: time series data (TSD)
    :param s_rate: sampling rate of TSD in Hz
    :param win: window fuction that will be applied on the data ("hanning" / no windowing)
    :return: [frequency], [Periodogram]
    '''
    winlength = data.size

    # define weight window (e.g Hanning Window)
    if win == "hanning":
        weight_win = np.hanning(winlength)

    else:
        weight_win = np.zeros(winlength) + 1

    # define power function window size and
    fft_win = int(np.floor(winlength / 2)) + 1

    # define fft frequency range
    powerfun_ticks = (np.arange(fft_win) + 1) / np.floor(winlength / s_rate)
    # do fft
    powerfun = 2 * np.fft.fft(data * weight_win) / winlength
    powerfun[0] = powerfun[0] / 2
    # calculate powert function
    powerfun = (pow(abs(powerfun), 2))[0:fft_win]
    return powerfun_ticks, powerfun


def mean_periodogram(data, s_rate=100, win_length=30, win="hanning"):
    '''
    calculates the periodogram of each window of a TSD and returns the mean energy per frequency as the
    final periodogram.
    :param data: time series data (TSD)
    :param sampling: sampling rate of TSD in Hz
    :param win: window fuction that will be applied on the data ("hanning" / no windowing)
    :param win_lenth: length of window in s
    :return: [frequency], [Periodogram]
    '''

    n_win_length = int(win_length * s_rate)
    fft_win = 1 + int(n_win_length / 2)
    n_windows = int(data.size / n_win_length)
    periodogram = np.zeros((n_windows, fft_win))
    for i in range(n_windows):
        start = n_win_length * i
        end = start + n_win_length
        fft_ticks, periodogram[i][:] = get_periodogram(data[start:end], s_rate, win)
    periodogram = np.mean(periodogram, axis=0)
    return fft_ticks, np.asarray(periodogram / n_windows)


def min_max_dist(data, s_rate, win_length=1):
    '''
    calculates max-min single distance of a function within a sliding window over a TSD. The value is mapped to the
    time value in the middle of the window.

    :param data: time series data (TSD)
    :param s_rate: sampling rate in Hz
    :param win_length: length of window in s
    :return: array of length of TSD with max-min distance per point
    '''

    N = data.size
    n_win_samp = win_length * s_rate
    d = int(n_win_samp / 2) + 1
    dist = np.zeros(N)
    for i in range(N):
        start = i - d
        end = i + d
        if start < 0:
            start = 0
        if end > N:
            end = N
        dist[i] = max(data[start:end]) - min(data[start:end])
    return dist


def mean_energy(data, s_rate=100, win_lengh=30):
    '''
    Tis function calculates the meaa energy of a signal per window

    :param data: time series data (TSD)
    :param s_rate: sampling rate in Hz
    :param win_lengh: window length in s
    :return: signal energy per window
    '''
    winleng = s_rate * win_lengh
    n_win = int(data / winleng)
    energy = np.zeros(n_win)
    for i in range(n_win):
        start = i * winleng
        end = start * winleng
        data_win = data[start:end]
        data_mean = np.mean(data_win)
        energy = sum((data_win - data_mean) ** 2) / winleng
    return energy


def energy_in_window(data, sampling, window=1, threshold=0.):
    '''
     calculates signal energy within a stepping window over a TSD.

     :param data: time series data (TSD)
     :param sampling: sampling rate in Hz
     :param window: length of window in s
     :param threshold:
     :return: array of length of TSD with max-min distance per point
     '''

    # set data < threshold to 0
    low_values_flags = (abs(data) < threshold)
    data[low_values_flags] = 0  # All low values set to 0

    # calculate window shape
    N = data.size
    window = int(window * sampling)
    n_windows = int(N / window)

    # calculate energy per window from i to i+window.
    sign_energy = np.zeros(N)
    for i in range(n_windows):
        start = int(i * window)
        end = int(start + window)
        if end > N:
            end = N
        sign_energy[start:end] = sum(pow(abs(data[start:end]), 2))
    return sign_energy


def envelope(data, sampling=1, threshold=0, min_length=0, norm=1):
    '''
     calculates the positive envelope of the absolute transform of a time series. Additionally, a threshold and a minimum
     length were the envelope is higher then the threshold can be applied. Values that do not satisfy these rules will
     be set to zero.

     inspired by: https://stackoverflow.com/questions/34235530/python-how-to-get-high-and-low-envelope-of-a-signal

    :param data: time series data
    :param sampling: sampling rate of TSD in hz
    :param threshold: minimum envelope of the signal(lower values will be set to zero)
    :param min_length: minimum time span that the envelope is > threshold (lower values will be set to zero)
    :return: [ envelope ]
    '''
    # optional normalisation of data
    if norm == "std":
        data = data  # / stat.stdev(data)
    if norm == "var":
        data = data / stat.var(data)
    elif (type(norm).__name__ == 'int') or (type(norm).__name__ == 'float'):
        data = data / norm


    # new envelope
    q = np.zeros(data.size)
    # transform all data to postive
    ts = abs(data)
    x = [0]
    y = [ts[0]]

    print("    ...find maximas...")
    for i in range(1, ts.size - 1):
        if np.sign(ts[i] - ts[i + 1]) == 1 and np.sign(ts[i] - ts[i - 1]) == 1:
            x.append(i)
            y.append(ts[i])
    # append the last value to the interpolating values
    x.append(ts.size - 1)
    y.append(ts[ts.size - 1])
    print("    ...fit model...")
    # fit a model to the data
    env_model = ip.interp1d(x, y, kind='cubic', bounds_error=False, fill_value=0.0)
    # estimate model points for ts
    q = np.asarray(env_model(np.linspace(1, ts.size, ts.size)))
    print("    ...apply threshold and min length...")
    # apply threshold
    flag = (q < threshold)
    q[flag] = 0

    # set to zero if length of signal > threshold is smaller than minimum length
    min_length = int(min_length * sampling)
    if (min_length > 1):
        i = 0
        while (i < data.size):
            # sore the current position as k
            k = i
            # expand k to the next position wee 0 value occurs
            while (q[i] > 0.) and (i < data.size):
                i = i + 1
            # if the distance between current position an k is smaller than min_length, erase the values
            if (int(i - k) < int(min_length)):
                q[k:i] = 0
            i = i + 1
    return np.asarray(q)


def pwspec_feat_extr(values, freq, sampling_rate):
    pass


def array_merge(a, b):
    out_array = []
    if len(a) != len(b):
        print('Error: Arrays must have equal length for merging')
        return False
    for i in range(0, len(a)):
        appender = [a[i], b[i]]
        out_array.append(appender)
    return out_array


def band_slicer(values):
    # 0-4 Hz
    delta = []
    # 4-8 Hz
    theta = []
    # 8-13 Hz
    alpha = []
    # 13-22
    beta = []
    # >30 Hz
    gamma = []
    for i in range(0, len(values)):
        frequency = values[i][0]
        if frequency <= 4:
            delta.append(values[i][1])
        elif frequency >= 4 and frequency <= 8:
            theta.append(values[i][1])
        elif frequency >= 8 and frequency <= 13:
            alpha.append(values[i][1])
        elif frequency >= 13 and frequency <= 22:
            beta.append(values[i][1])
        elif frequency >= 22 and frequency <= 30:
            gamma.append(values[i][1])
    return delta, theta, alpha, beta, gamma


def get_hypno_array(path):
    hypno_array = []
    with open(path, 'rU') as csvfile:
        print("Read hypnogram from: ", path)
        reader = csv.reader(csvfile)
        for line in reader:
            hypno_array.append(line[0])
    return hypno_array


def data2file(fb, data, hypno_array=False, win_length=1, s_rate=1):
    n_win_length = win_length * s_rate
    n_features = data.shape[0] + 3
    n_win = data.shape[1]
    appender = np.zeros(n_features)
    for i in range(n_win):
        appender[0] = i * n_win_length  # start of time window
        appender[1] = appender[0] + n_win_length - 1  # end of time window
        if hypno_array == False:
            appender[2] = np.nan  # hypnogram as nan
        else:
            appender[2] = hypno_array[i]  # hypnogram from file
        appender[3:n_features] = data[:][i]  # data
        fb.write(str(appender) + os.linesep)  # write line to file
