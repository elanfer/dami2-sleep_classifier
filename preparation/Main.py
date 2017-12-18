# This packages might be needed
"""""
 python-vtk
 python -mip install -U pip
 pip install numpy
 pip install scipy
 pip install mayavi
 pip install PySurfer mne
 python -mpip install -U matplotlib
"""
import numpy as np

import feature_extraction as extract
import tools

if __name__ == '__main__':
    # _________________________________________________________________________________________________________
    # define basic parameters

    # can also be a text file or list of paths
    filePath = "../SC4001E0/SC4001E0-PSG.edf"
    hypPath = "../SC4001E0"
    hypPath_csv = '../SC4001E0/SC4001E0-PSG.edf.csv'
    # start at step (in seconds)
    start = 0
    # use time series data from start until n-windows further (windows of the length win_length in seconds)
    n_windows = np.nan
    # classification window length in seconds
    win_length = 30
    # sampling rate
    s_rate = 100

    # calculate end-position (can handle nan and too long inputs => sets them to max size )
    start = start * win_length * s_rate
    end = start + n_windows * win_length * s_rate

    # eeg frequ. [start,end]:  [ [delta]    [theta]      [alpha]       [beta]        [gamma] ]
    eeg_freq = np.asarray([[.16, 4.0], [4.0, 8.0], [8.0, 13.0], [13.0, 22.0], [22.0, 45.0]])
    # eog frequ.
    eog_freq = np.asarray([[0.1, 0.3], [0.35, 0.5]])
    # spindle frequency range:
    spindle_freq = np.asarray([11.5, 14.5])


    # _____________________________________________________________________________________________________
    # load hypnogram
    hypno_array, hypnogram = extract.convert_hypno(hypPath_csv=hypPath_csv, s_rate=s_rate, win_length=win_length)

    # load data
    ts = extract.load_data(filePath, hypnogram=hypnogram)

    # get basic properties as plots, calculate spindle envelope
    # [spindle_env]
    spindle_env = extract.basic_properties(ts, hypnogram,
                                           s_rate=s_rate,
                                           win_length=win_length,
                                           eeg_freq=eeg_freq,
                                           spindle_freq=spindle_freq,
                                           show_properties=False)

    ts_energy = extract.feature_extraction(ts, spindle_env,
                                           start=start,
                                           end=end,
                                           s_rate=s_rate,
                                           win_length=win_length,
                                           eeg_freq=eeg_freq,
                                           eog_freq=eog_freq)

    # ____________________________________________________________________________________________________________
    # write data to file
    # current structure of text-file:
    # from[1]  to[1]  SleepStage[1]  eeg-features[6]  eog-features[2]  emg-features[1]   (= 11 entries per line)
    with open('../testrun2.txt', 'a') as fp:
        tools.data2file(fp, ts_energy,
                        hypno_array=hypno_array,
                        win_length=win_length,
                        s_rate=s_rate)
