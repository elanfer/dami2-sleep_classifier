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

import SleepData as sd

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

file.preprocess()
sleep_val = file.get_data(100, 0, 1000000)
eeg = file.get_eeg(100, 0, 10000)
emg = []

for i in range(0, len(sleep_val)):
    for j in range(0, len(sleep_val[i])):
        emg.append(sleep_val[i][j][1])

eog = []
for i in range(0, len(sleep_val)):
    for j in range(0, len(sleep_val[i])):
        eog.append(sleep_val[i][j][1])

plt.plot(eeg)
plt.ylabel('frequency')
plt.xlabel('Time')
plt.show()
