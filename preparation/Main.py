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
import irasa
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
    start = 4000000
    winlength = 3000
    sampling = 100

    # get data
    ts = np.asarray(file.get_eeg(start, start + winlength))
    # do pre processing
    # ts = tools.butter_bandpass_filter(ts, 0.15, 100)
    h = np.linspace(1.1, 1.9, num=9)

    test = irasa.irisa(ts, sampling)

    # new_ts = tools.random_sampling_ts(ts)
    fft_ticks, powerfun = tools.powerfunction(ts, 100, False)
    fft_ticks, rand_powerfun = tools.random_powerfunction(ts, 100, False, iter=10)
    powerdiff = 1 + powerfun - rand_powerfun



    # create plot
    plt.figure(2)
    plt.subplot(211)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(test[2][:], test[0][:])
    plt.plot(test[2][:], test[1][:])

    plt.subplot(212)

    plt.xscale('log')
    plt.yscale('log')
    plt.plot(fft_ticks, powerfun)
    plt.plot(test[2][:], test[0][:])
    plt.show()
