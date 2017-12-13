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
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat

import SleepData as sd
import amri_sig_filtfft as amri
import tools

if __name__ == '__main__':
    # can also be a text file or list of paths
    filePath = "../SC4001E0/SC4001E0-PSG.edf"
    hypPath = "../SC4001E0"

    # concerting the given hypnograms to a csv:
    # hypno.convert_hypnograms(hypPath)

    # read sleep data
    file = sd.SleepData(filePath,
                        # start = ,
                        # stop = ,
                        channels={'EEG': 'EEG Fpz-Cz', 'EMG': 'EMG submental', 'EOG': 'EOG horizontal'},
                        # preload=False
                        )

    # define size of time window
    start = 5000000  # 2500000 +2000000
    windows = 100
    winlength = 3000 * windows
    sampling = 100

    # get data
    ts = np.asarray(file.get_eeg(start, start + winlength))

    #### do pre processing:
    # filtering
    filt = amri.amri_sig_filtfft(ts, fs=100, lowcut=12, highcut=14, trans=0.015, revfilt=True)
    # -- Normalize signal
    # filt = filt/stat.stdev(filt)
    # calculate threshold
    filt_std_th = 2.0 * stat.stdev(filt)
    # calculate min-max distance and signal energy
    # energy = tools.energy_in_window(filt,100,window=30)
    # dist = tools.min_max_dist(filt,100,window = 0.5)
    # calculate envelope
    envelope = tools.get_envelope(filt, sampling=100, threshold=filt_std_th, min_length=1)

    # calculate periodogram:
    fft_ticks, powerfun = tools.powerfunction(ts, 100)
    # mean periodogram
    fft_ticks_mean, mean_powerfun = tools.mean_periodogram(ts, 100, windowing=windows)
    # calculate mean periodogram of filtered TS
    fft_ticks_mean_filt, mean_powerfun_filt = tools.mean_periodogram(filt, 100, windowing=windows)
    # envelope periodogrma
    fft_ticks_mean_env, mean_powerfun_env = tools.mean_periodogram(envelope, 100, windowing=windows)

    plt.figure(0)
    plt.hist(envelope, 150)
    plt.show()

    plt.figure(1)
    plt.plot(ts)
    plt.plot(envelope)
    plt.show()

    # create plot
    plt.figure(2)
    plt.subplot(211)
    # plt.xscale('log')
    plt.yscale('log')
    plt.plot(fft_ticks_mean, mean_powerfun)
    plt.plot(fft_ticks_mean_filt, mean_powerfun_filt)

    plt.subplot(212)
    # plt.xscale('log')
    plt.yscale('log')
    plt.plot(fft_ticks_mean_env, mean_powerfun_env)
    plt.show()
