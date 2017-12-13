import csv
import numpy as np
import scipy.interpolate as ip
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
    '''
    function to calculate the periododogram of a TSD

    :param data: time series data (TSD)
    :param sampling: sampling rate of TSD in Hz
    :param win: window fuction that will be applied on the data ("hanning" / no windowing)
    :return: [frequency], [Periodogram]
    '''
    winlength = len(data)

    # define weight window (e.g Hanning Window)
    if win == "hanning":
        weight_win = np.hanning(winlength)

    else:
        weight_win = np.zeros(winlength) + 1

    # define power function window size and
    fft_win = int(np.floor(winlength / 2)) + 1

    # define fft frequency range
    powerfun_ticks = (np.arange(fft_win) + 1) / np.floor(winlength / sampling)
    # do fft
    powerfun = 2 * np.fft.fft(data * weight_win) / winlength
    powerfun[0] = powerfun[0] / 2
    # calculate powert function
    powerfun = (pow(abs(powerfun), 2))[0:fft_win]
    return powerfun_ticks, powerfun


def mean_periodogram(data, sampling, win="hanning", windowing=3):
    '''
    calculates the periodogram of each window of a TSD and returns the mean energy per frequency as the
    final periodogram.
    :param data:
    :param sampling:
    :param win:
    :param windowing:
    :return: [frequency], [Periodogram]
    '''
    step = int(len(data) / windowing)
    fft_win = 1 + int(step / 2)
    windowing = int(windowing)
    periodogram = np.zeros((windowing, fft_win))
    for i in range(windowing):
        start = windowing * i
        end = start + step
        fft_ticks, periodogram[i][:] = powerfunction(data[start:end], sampling, win)
    periodogram = np.mean(periodogram, axis=0)
    return fft_ticks, np.asarray(periodogram / windowing)


def min_max_dist(data, sampling, window=1, threshold=0.):
    '''
    calculates max-min single distance of a function within a sliding window over a TSD. The value is mapped to the
    time value in the middle of the window.

    :param data: time series data (TSD)
    :param sampling: sampling rate in Hz
    :param window: length of window in s
    :param threshold:
    :return: array of length of TSD with max-min distance per point
    '''

    # set data < threshold to 0
    low_values_flags = (abs(data) < threshold)
    data[low_values_flags] = 0  # All low values set to 0

    N = data.size
    window = window * sampling
    d = int(window / 2) + 1
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

    # inspired by: https://stackoverflow.com/questions/34235530/python-how-to-get-high-and-low-envelope-of-a-signal


def get_envelope(data, sampling=1, threshold=0, min_length=0):
    '''
     calculates the positive envelope of the absolute transform of a time series. Additionally, a threshold and a minimum
     length were the envelope is higher then the threshold can be applied. Values that do not satisfy these rules will
     be set to zero.

    :param data: time series data
    :param sampling: sampling rate of TSD in hz
    :param threshold: minimum envelope of the signal(lower values will be set to zero)
    :param min_length: minimum time span that the envelope is > threshold (lower values will be set to zero)
    :return: [ envelope ]
    '''
    # new envelope
    q = np.zeros(data.size)

    # transform all data to posive
    ts = abs(data)

    x = [0]
    y = [ts[0]]

    for i in range(1, ts.size - 1):
        if np.sign(ts[i] - ts[i + 1]) == 1 and np.sign(ts[i] - ts[i - 1]) == 1:
            x.append(i)
            y.append(ts[i])

    # append the last value to the interpolating values
    x.append(ts.size - 1)
    y.append(ts[ts.size - 1])

    # fit a model to the data
    env_model = ip.interp1d(x, y, kind='cubic', bounds_error=False, fill_value=0.0)
    # estimate model points for ts
    q = np.asarray(env_model(np.linspace(1, ts.size, ts.size)))
    # apply threshold
    flag = (q < threshold)
    q[flag] = 0

    # set to zero if length of signal > threshold is smaller than minimum length
    min_length = int(min_length * sampling)
    if (min_length > sampling):
        for i in range(data.size):
            # sore the current position as k
            k = i
            # expand k to the next position wee 0 value occurs
            while (q[k] > 0.) and (k < data.size):
                k = k + 1
            # if the distance between current position an k is smaller than min_length, erase the values
            if (int(k - i) < int(min_length)):
                q[i:k] = 0
            # set the current position to k
            i = k
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
        reader = csv.reader(csvfile)
        for line in reader:
            hypno_array.append(line[0])
    return hypno_array
