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

    eeg = file.get_eeg(0, 6000)
    print (file.info())
    eeg = tools.butter_bandpass_filter(eeg, 0.15, 100)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(eeg)

    eeg = np.fft.fft(eeg)
    plt.subplot(212)
    geloet = (pow(abs(eeg), 2) + np.hanning(6000))
    geloet = geloet[0:3000]
    x_ticks_real = range(0, 3000, 200)
    x_ticks_mod = []
    for i in (range(0, len(x_ticks_real))):
        x_ticks_mod.append(x_ticks_real[i] / 60)
    print (x_ticks_mod)
    plt.xticks(x_ticks_real, x_ticks_mod)
    plt.yscale('log')
    plt.plot(geloet)
    plt.show()
