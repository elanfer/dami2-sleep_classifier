# This packages might be needed
"""""
 python -mip install -U pip
 pip install numpy
 pip install scipy
 pip install PySurfer mne
 python -mpip install -U matplotlib
"""
import SleepData
import convert_hypnograms

# can also be a text file or list of paths
filePath = "../SC4001E0/SC4001E0-PSG.edf"
hypPath = "../SC4001E0"

# concerting the given hypnograms to a csv:
convert_hypnograms(hypPath)

# read sleep data
file = SleepData(filePath,
                 # start = ,
                 # stop = ,
                 channels={'EEG': 'EEG Fpz-Cz', 'EMG': 'EMG submental', 'EOG': 'EOG horizontal'},
                 # preload=False
                 )

hypPath = "../SC4001E0/SC4001E0-PSG.csv"
