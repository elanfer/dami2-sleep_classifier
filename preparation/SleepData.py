# -*- coding: utf-8 -*-
"""
This function was provided by Simon Kern as part of the
[AutoSleepScorer project](https://github.com/skjerns/AutoSleepScorer)
to read in and work with .edf PSG-Data.

"""

import os
import re
from copy import deepcopy
from multiprocessing import Pool

import mne
import numpy as np


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


class SleepData(object):
    def __init__(self, file, preload=True, channels=None, references=None,
                 epoch_len=3000, start=None, stop=None, use_mp=True):
        """
        The SleepData object can load an EEG file or header and perform the necessary preprocessing
        and channel selection.

        :param file: a file string pointing to a sleep EEG header
        :param preload: load eeg while instantiating. Else data will be loaded if needed
        :param channels: A dictionary in the following format, where the value corresponds to
                         the name of the cannel of that modality {'EEG':'C4', 'EMG':'submental', 'EOG':'LeftEye'}
        :param references: A dictionary in the following format, where the value corresponds to
                         the name of the reference for the modality {'RefEEG':'C4', 'RefEMG':False, 'RefEOG':'LeftEye'}
        :param epoch_len: Epoch length in samples. Not supported yet.
        :param start: The sample at which to start classification
        :param stop: The sample at which to stop classification (length will be trimmed to a multiple of epoch_len)
        :param use_mp: use multiprocessing. Will instantiate 3 workers usable by all SleepData instances.
        """
        if not os.path.isfile(file): raise FileNotFoundError('File {} not found'.format(file))
        if use_mp and not hasattr(SleepData, 'pool'): SleepData.pool = Pool(3)
        self.header = None
        self.epoch_len = epoch_len
        self.start = start
        self.stop = stop
        self.use_mp = use_mp
        self.available_channels = []
        self.file = file
        self.loaded = False
        self.printed_channels = False
        self.channels = {'EEG': False, 'EMG': False, 'EOG': False} if channels is None else channels
        self.references = {'RefEEG': False, 'RefEMG': False, 'RefEOG': False} if references is None else references
        self.epoch_len = epoch_len

        if preload == True:
            self.load()

    def _print(self, string):
        print(string)

    def infer_channels(self, channels, ch_type='all'):
        """
        Tries to automatically infer channel names. Very limited functionality.

        :param channels: a list of channel names
        :param ch_type: The type of channel that you want to infer (EEG, EMG, EOG or all)
        :returns: tuple(channel, reference) if one channel, dictionary with mappings if all channels
        """
        if not self.printed_channels:
            self._print('Available channels: ' + str(channels))
            self.printed_channels = True

        channels = [c.upper() for c in channels]

        def infer_eeg(channels):
            # Infer EEG
            ch = False
            ref = False
            if 'EEG' in channels:
                ch = 'EEG'
            elif 'C3' in channels and 'A2' in channels:
                ch = 'C3'
                ref = 'A2'
            elif 'C4' in channels and 'A1' in channels:
                ch = 'C4'
                ref = 'A1'
            elif 'FPZ' in channels and 'CZ' in channels:
                ch = 'FPZ'
                ref = 'CZ'
            elif 'PZ' in channels and 'OZ' in channels:
                ch = 'PZ'
                ref = 'OZ'
            else:
                for c in channels:
                    if 'C4' in c and 'A1' in c:
                        ch = c;
                        break
                    if 'C3' in c and 'A2' in c:
                        ch = c;
                        break
                    if 'EEG' in c:
                        ch = c;
                        break
            self._print('Infering EEG Channel... {}, Ref: {}'.format(ch, ref))
            return ch, ref

        def infer_emg(channels):
            ch = False
            ref = False
            if 'EMG' in channels:
                ch = 'EMG'
                ref = False
            elif 'EMG1' in channels and 'EMG2' in channels:
                ch = 'EMG1'
                ref = 'EMG2'
            else:
                for c in channels:
                    if 'EMG' in c:
                        ch = c
                        break
            self._print('Infering EMG Channel... {}, Ref: {}'.format(ch, ref))
            return ch, ref

        def infer_eog(channels):
            ch = False
            ref = False
            if 'EOG' in channels:
                ch = 'EOG'
            elif 'LOC' in channels:
                ch = 'LOC'
            elif 'ROC' in channels:
                ch = 'ROC'
            elif 'EOG horizontal' in channels:
                ch = 'EOG HORIZONTAL'
            else:
                for c in channels:
                    if 'EOG' in c or 'EYE' in c:
                        ch = c
                        break
            self._print('Infering EOG Channel... {}, Ref: {}'.format(ch, ref))
            return ch, ref

        if ch_type.upper() == 'EEG':   return infer_eeg(channels)
        if ch_type.upper() == 'EMG':   return infer_emg(channels)
        if ch_type.upper() == 'EOG':   return infer_eog(channels)
        if ch_type.lower() == 'all':
            eeg, refeeg = infer_eeg(channels)
            emg, refemg = infer_emg(channels)
            eog, refeog = infer_eog(channels)
            return ({'EEG': eeg, 'EMG': emg, 'EOG': eog},
                    {'RefEEG': refeeg, 'RefEMG': refemg, 'RefEOG': refeog})
        raise Exception('Infer_channels: Wrong channel type selected: {}'.format(ch_type))

    def load_eeg_header(self, filename=None, dataformat='', **kwargs):  # CHECK include kwargs
        """
        loads an EEG header using MNE given a filename.
        Tries to automatically infer the data format


        :param channels: a list of channel names
        :param ch_type: The type of channel that you want to infer (EEG, EMG, EOG or all)
        :returns: tuple(channel, reference) if one channel, dictionary with mappings if all channels
        """
        if filename is None: filename = self.file
        dataformats = dict({
            # 'bin' :'artemis123',
            '???': 'bti',  # CHECK
            'cnt': 'cnt',
            'ds': 'ctf',
            'edf': 'edf',
            'bdf': 'edf',
            'sqd': 'kit',
            'data': 'nicolet',
            'set': 'eeglab',
            'vhdr': 'brainvision',
            'egi': 'egi',
            'fif': 'fif',
            'gz': 'fif',
        })

        if dataformat == '':  # try to guess format by extension
            ext = os.path.splitext(filename)[1][1:].strip().lower()
            dataformat = dataformats[ext]

        if not 'verbose' in kwargs: kwargs['verbose'] = 0

        if dataformat == 'artemis123':
            header = mne.io.read_raw_artemis123(filename, verbose=0, **kwargs)  # CHECK if now in stable release
        elif dataformat == 'bti':
            header = mne.io.read_raw_bti(filename, **kwargs)
        elif dataformat == 'cnt':
            header = mne.io.read_raw_cnt(filename, **kwargs)
        elif dataformat == 'ctf':
            header = mne.io.read_raw_ctf(filename, **kwargs)
        elif dataformat == 'edf':
            kwargs['stim_channel'] = None
            header = mne.io.read_raw_edf(filename, **kwargs)
        elif dataformat == 'kit':
            header = mne.io.read_raw_kit(filename, **kwargs)
        elif dataformat == 'nicolet':
            header = mne.io.read_raw_nicolet(filename, **kwargs)
        elif dataformat == 'eeglab':
            header = mne.io.read_raw_eeglab(filename, **kwargs)
        elif dataformat == 'brainvision':  # CHECK NoOptionError: No option 'markerfile' in section: 'Common Infos'
            header = mne.io.read_raw_brainvision(filename, **kwargs)
        elif dataformat == 'egi':
            header = mne.io.read_raw_egi(filename, **kwargs)
        elif dataformat == 'fif':
            header = mne.io.read_raw_fif(filename, **kwargs)
        else:
            raise FileNotFoundError('Failed extension not recognized for file: '.format(filename))
        if not 'verbose' in kwargs: print('loaded header ' + filename)

        self.sfreq = np.round(header.info['sfreq'])
        self.header = header
        return header

    def pick_channels(self):
        self.available_channels = self.header.ch_names
        if self.channels['EEG'] == False: self.channels['EEG'], self.references['RefEEG'] = self.infer_channels(
            self.header.ch_names, 'EEG')
        if self.channels['EMG'] == False: self.channels['EMG'], self.references['RefEMG'] = self.infer_channels(
            self.header.ch_names, 'EMG')
        if self.channels['EOG'] == False: self.channels['EOG'], self.references['RefEOG'] = self.infer_channels(
            self.header.ch_names, 'EOG')

        channels = [c.upper() for c in self.header.ch_names]
        filename = self.file
        labels = []
        picks = []
        for ch in self.channels:
            if self.channels[ch].upper() not in channels:
                raise ValueError(
                    'ERROR: Channel {} for {} not found in {}\navailable channels: {}'.format(self.channels[ch], ch,
                                                                                              filename, channels))
            else:
                picks.append(channels.index(self.channels[ch].upper()))
                labels.append(ch)
        for ch in self.references:
            if not self.references[ch]: continue
            if self.references[ch] not in channels:
                raise ValueError(
                    'ERROR: Channel {} for {} not found in {}\navailable channels: {}'.format(self.references[ch], ch,
                                                                                              filename, channels))
            else:
                picks.append(channels.index(self.references[ch]))
                labels.append(ch)
        self.channel_ids = (picks, labels)
        return (picks, labels)

    def load_data(self):
        picks, labels = self.channel_ids
        data, _ = deepcopy(self.header[picks, :])
        # rereferencing
        self.eeg = data[labels.index('EEG'), :]
        if self.references['RefEEG']:
            self.eeg = self.eeg - data[self.channel_ids[1].index('RefEEG'), :]
        self.emg = data[labels.index('EMG'), :]
        if self.references['RefEMG']:
            self.emg = self.emg - data[self.channel_ids[1].index('RefEMG'), :]
        self.eog = data[labels.index('EOG'), :]
        if self.references['RefEOG']:
            self.eog = self.eog - data[self.channel_ids[1].index('RefEOG'), :]

    def preprocess(self):

        # resampling
        if not np.isclose(self.sfreq, 100):
            print('resampling from {} Hz to {} Hz'.format(self.sfreq, 100))
            if self.use_mp and hasattr(SleepData, 'pool') and type(SleepData.pool) is Pool:
                res_eeg = SleepData.pool.apply_async(
                    mne.io.RawArray(np.stack([self.eeg]), mne.create_info(1, self.sfreq, 'eeg'), verbose=0).resample,
                    args=(100.,))
                res_emg = SleepData.pool.apply_async(
                    mne.io.RawArray(np.stack([self.emg]), mne.create_info(1, self.sfreq, 'eeg'), verbose=0).resample,
                    args=(100.,))
                res_eog = SleepData.pool.apply_async(
                    mne.io.RawArray(np.stack([self.eog]), mne.create_info(1, self.sfreq, 'eeg'), verbose=0).resample,
                    args=(100.,))
                eeg, _ = res_eeg.get(timeout=30)[0, :]
                emg, _ = res_emg.get(timeout=30)[0, :]
                eog, _ = res_eog.get(timeout=30)[0, :]

            else:
                eeg, _ = mne.io.RawArray(np.stack([self.eeg]), mne.create_info(1, self.sfreq, 'eeg'),
                                         verbose=0).resample(100.)[0, :]
                emg, _ = mne.io.RawArray(np.stack([self.emg]), mne.create_info(1, self.sfreq, 'eeg'),
                                         verbose=0).resample(100.)[0, :]
                eog, _ = mne.io.RawArray(np.stack([self.eog]), mne.create_info(1, self.sfreq, 'eeg'),
                                         verbose=0).resample(100.)[0, :]

            self.sfreq = 100
            self.eeg = eeg.squeeze()
            self.emg = emg.squeeze()
            self.eog = eog.squeeze()

    def load(self, file=None):
        if file is None: file = self.file
        self.load_eeg_header(file, verbose='WARNING')
        self.pick_channels()
        self.load_data()
        self.preprocess()
        self.loaded = True
        return self.get_data()

    def get_data(self, epoch_len=None, start=None, stop=None):
        """
        :param epoch_len: Select length of epoch in samples with sampling frequency 100. Default: 3000
        :param start: select starting point of selection in samples with sfreq=100
        :param stop: select stopping point of selection in samples with sfreq=100
        """
        if start: self.start = start
        if stop: self.stop = stop
        if epoch_len: self.epoch_len = epoch_len
        if not self.loaded:
            self.load()
        signal = np.vstack([self.eeg, self.emg, self.eog]).swapaxes(0, 1)
        mean = np.mean(signal)
        std = np.std(signal)
        signal = signal[self.start:self.stop]
        signal = signal[:len(signal) - len(signal) % self.epoch_len]
        signal = signal.reshape([-1, self.epoch_len, 3])
        signal = signal - mean
        signal = signal / std
        return signal.astype(np.float32)

    def info(self):
        memsize = '{:.1f}'.format(
            (self.eeg.nbytes + self.eog.nbytes + self.emg.nbytes) / 1024 ** 2) if self.loaded else 'Not loaded'
        print('File:      {} ({:.1f} MB)'.format(self.file, os.path.getsize(self.file) / 1024 ** 2))
        print('Memory:    {} MB'.format(memsize))
        print('Epoch len: {} samples'.format(self.epoch_len))
        print('Selection: {} to {}'.format(self.start, self.stop))
        print('Channel Information')
        print('Available channels: {}'.format(self.available_channels))
        print('Selected channels:  {}'.format(self.channels))
        print('Selected references: {}'.format(self.references))

    def __call__(self):
        return self.get_data()

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, start, stop, step):
        return self.get_data(start=start, stop=stop)


# %%

class SleepDataset(object):
    def __init__(self, directory):
        """
        :param directory: a directory string
        """
        pass
