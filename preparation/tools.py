import csv

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
