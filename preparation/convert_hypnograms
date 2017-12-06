"""
The convert_hypnograms function is a hack that searches for the string <code>Sleep\_stage\_</code>
and extracts the subsequently character.
It was gratefully provided by [Simon Kern](https://pastebin.com/Qx2qbQVP) due to the absence ob a proper
python packages to load hypnogram data from the [PhysioNet data base](https://physionet.org/pn4/eegmmidb/).
"""

import csv
import os


def convert_hypnograms(datadir):
    """
    This function is quite a hack to read the edf hypnogram as a byte array.
    I found no working reader for the hypnogram edfs.
    """
    print("Converting all .edf hypnograms in this directory:\n   ", datadir,
          "\nto separate .csv-files.")
    files = [x for x in os.listdir(datadir) if x.endswith('.hyp')]
    for file in files:
        file = os.path.join(datadir, file)
        hypnogram = []
        with open(file, mode='rb') as f:  # b is important -> binary

            raw_hypno = [x for x in str(f.read()).split('Sleep_stage_')][1:]
            for h in raw_hypno:
                stage = h[0]
                repeat = int(h.split('\\')[0][12:]) // 30  # no idea if this also works on linux
                hypnogram.extend(stage * repeat)
        with open(file[:-4] + '.csv', "w") as f:
            writer = csv.writer(f, lineterminator='\r')
            writer.writerows(hypnogram)
