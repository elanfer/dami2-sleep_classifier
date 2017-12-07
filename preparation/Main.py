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

import SleepData as sd
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
    winlength = 3000
    sampling = 100

    # get data
    ts = file.get_eeg(0, winlength)
    # do pre processing
    ts = tools.butter_bandpass_filter(ts, 0.15, 100)


    def powerfunction(data, sampling, win="hanning"):

        winlength = len(data)

        # define weight window (e.g Hanning Window)
        if win == "hanning":
            weight_win = np.hanning(winlength)

        # define power function window size and
        fft_win = int(np.floor(winlength / 2))
        # define fft frequency range
        fft_ticks = np.arange(1, fft_win + 1, 1) / np.floor(winlength / sampling)
        # do fft
        fft_result = np.fft.fft(data)
        plt.subplot(212)
        # calculate powert function
        powerfun = (pow(abs(fft_result), 2) * weight_win)[0:fft_win]
        return fft_ticks, powerfun


    # create plot
    plt.figure(1)
    plt.subplot(211)
    plt.plot(ts)
    hypno_array = tools.get_hypno_array('../SC4001E0/SC4001E0-PSG.edf.csv')
    print len(hypno_array)
    fft_ticks, powerfun = powerfunction(ts, 100)
    power_mat = tools.array_merge(fft_ticks, powerfun)
    delta, theta, alpha, beta, gamma = tools.band_slicer(power_mat)
    plt.yscale('log')
    plt.plot(fft_ticks, powerfun)
    plt.show()
