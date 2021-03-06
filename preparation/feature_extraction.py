import math
import os.path

import numpy as np

import SleepData as sd
import amri_sig_filtfft as amri
import convert_hypnograms
import tools


def load_data(file_path, hypnogram):
    '''
    function to load signal and hypnogram from the data file and store them all in the same length

    :param file_path: path of data file
    :param hypnogram: hypnogram upsampled
    :return: [EEG-data],[EOG-data],[EMG-data],[hypnogram]
    '''

    # read sleep data
    file = sd.SleepData(file_path, channels={'EEG': 'EEG Fpz-Cz', 'EMG': 'EMG submental', 'EOG': 'EOG horizontal'})
    start = 0
    end = len(hypnogram)

    ts = np.asarray([np.asarray(file.get_eeg(start, end)),
                     np.asarray(file.get_eog(start, end)),
                     np.asarray(file.get_emg(start, end))])
    return ts


def basic_properties(ts, hypnogram,
                     s_rate=100,
                     win_length=30,
                     eeg_freq=np.asarray([[.16, 4.], [4., 8.], [8., 13.], [13., 22.], [22., 50.]]),
                     spindle_freq=np.asarray([11.5, 14.5]),
                     show_properties=True):
    # calculate window length
    n_win_length = s_rate * win_length

    # filter the data in their specific frequency
    print("do filtering... ")
    # deta approx. 0-4 Hz
    delta = []
    # delta = amri.amri_sig_filtfft(eeg_ts, fs= s_rate, lowcut= eeg_freq[0][0], highcut=eeg_freq[0][1], trans=0.015, revfilt=True)
    # theta approx. 4-8 Hz
    theta = []
    # theta = amri.amri_sig_filtfft(eeg_ts, fs= s_rate, lowcut= eeg_freq[1][0], highcut=eeg_freq[1][1], trans=0.015, revfilt=True)
    # alpha approx. 8-13 Hz
    alpha = []
    # alpha = amri.amri_sig_filtfft(eeg_ts, fs= s_rate, lowcut= eeg_freq[2][0], highcut=eeg_freq[2][1], trans=0.015, revfilt=True)
    # beta approx. 13-22 Hz
    beta = []
    # beta = amri.amri_sig_filtfft(eeg_ts, fs= s_rate, lowcut= eeg_freq[3][0], highcut=eeg_freq[3][1], trans=0.015, revfilt=True)
    # gamma approx. >30 Hz
    gamma = []
    # gamma = amri.amri_sig_filtfft(eeg_ts, fs= s_rate, lowcut= eeg_freq[4][0], highcut=eeg_freq[4][1], trans=0.015, revfilt=True)

    # spindle signal preprocessing:
    # approx. 11,5 - 14,5 Hz
    spindle = amri.amri_sig_filtfft(ts[0], fs=s_rate, lowcut=spindle_freq[0], highcut=spindle_freq[1], trans=0.015,
                                    revfilt=True)
    print("      ...finished.")
    print("calculate envelopes...")
    # calculate threshold:
    spindle_threshold = 2.0 * tools.sampled_std(spindle, max=1000000)


    # calculate envelope with minimum lenght of 0.5 seconds
    spindle_env = tools.envelope(spindle, sampling=s_rate, threshold=spindle_threshold, min_length=0.5, norm=False)

    # teta envelope
    # theta_env = tools.envelope(theta, sampling=s_rate, norm = "std" )
    print("      ...finished.")


    return spindle_env


def feature_extraction(ts, spindle_env,
                       start=0,
                       end=np.nan,
                       s_rate=100,
                       win_length=30,
                       eeg_freq=np.asarray([[.16, 4.], [4., 8.], [8., 13.], [13., 22.], [22., 50.]]),
                       eog_freq=np.asarray([[0.1, 0.3], [0.35, 0.5]])):

    print("extract features...")
    # cut TSD to defined range
    start = int(start * s_rate * win_length)
    if math.isnan(end) or end > ts.shape[1]:
        end = int(ts.shape[1])
        print("set end value to :", end)
    else:
        end = int(end)

    ts = ts[:][start:end]
    spindle_env = spindle_env[start:end]

    # calculate window length
    n_win_length = s_rate * win_length
    n_win = int(ts.shape[1] / n_win_length)

    eeg_energy = np.zeros([eeg_freq.shape[0] + 1, n_win])
    eog_energy = np.zeros([eog_freq.shape[0], n_win])
    emg_energy = np.zeros(n_win)
    periodograms = np.zeros([ts.shape[0], int(n_win_length / 2 + 1)])

    for i in range(n_win):
        start = i * n_win_length
        end = start + n_win_length
        # calculate periodograms per window
        for j in range(ts.shape[0]):
            ticks, periodograms[j][:] = tools.get_periodogram(ts[j][start:end], s_rate)

        # calculate eeg energy per frequency domain
        for k in range(eeg_freq.shape[0]):
            eeg_energy[k, i] = periodograms[0][
                np.logical_and((ticks >= eeg_freq[k][0]), (ticks <= eeg_freq[k][1]))].sum()
        eeg_energy[eeg_freq.shape[0], i] = pow((spindle_env[start:end] - np.mean(spindle_env[start:end])),
                                               2).sum() / n_win_length
        for k in range(eog_freq.shape[0]):
            eog_energy[k, i] = periodograms[1][
                np.logical_and((ticks >= eog_freq[k][0]), (ticks <= eog_freq[k][1]))].sum()
        emg_energy[i] = pow((ts[2][start:end] - np.mean(ts[2][start:end])), 2).sum() / n_win_length

    ts_energy = np.asarray([eeg_energy, eog_energy, emg_energy])
    print("      ...finished.")

    return ts_energy


def convert_hypno(hypPath=False, hypPath_csv=False, s_rate=1, win_length=1):
    '''
    get hypnogram with sample size of data
       Stages:
               Wake    = 0
               S1      = 1
               S2      = 2
               S3      = 3
               S4      = 4
               REM     = 5
    :param hypPath: path of original hypnogram (if converting in necessary)
    :param hypPath_csv:  path of csv hypnogram (if not a csv, convert it)
    :param s_rate: sampling rate in Hz
    :param win_length: length of window in s
    :return: [original hypnogram][high sampled hypnogram]
    '''
    # load hypnogram
    n_win_samp = s_rate * win_length

    if (os.path.exists(hypPath_csv)):
        hypno_array = tools.get_hypno_array(hypPath_csv)
    elif (os.path.exists(hypPath)):
        # concerting the given hypnograms to a csv:
        convert_hypnograms.convert_hypnograms(hypPath)
        hypno_array = tools.get_hypno_array(hypPath_csv)
    else:
        print (" no hypnogram file available!")
        exit()

    hypnogram = np.zeros(len(hypno_array) * n_win_samp)
    for i in range(len(hypno_array)):
        hypno_val = hypno_array[i]
        if hypno_val == "W": hypno_val = "0"
        if hypno_val == "R": hypno_val = "5"
        start = i * s_rate * win_length
        end = start + s_rate * win_length
        hypnogram[start:end] = ord(hypno_val) - 48
        hypno_array[i] = ord(hypno_val) - 48
    return hypno_array, hypnogram
