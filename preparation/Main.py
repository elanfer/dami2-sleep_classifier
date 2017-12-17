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
    start = 0
    n_windows = 10
    win_length = 30
    s_rate = 100
    end = start + n_windows * win_length * s_rate
    # eeg frequ. [start,end]:  [ [delta]    [theta]      [alpha]       [beta]        [gamma] ]
    eeg_freq = np.asarray([[.16, 4.0], [4.0, 8.0], [8.0, 13.0], [13.0, 22.0], [22.0, 45.0]])
    # spindle frequency range:
    spindle_freq = np.asarray([11.5, 14.5])

    # _____________________________________________________________________________________________________
    # load hypnogram
    hypno_array, hypnogram = extract.convert_hypno(hypPath_csv=hypPath_csv, s_rate=s_rate, win_length=win_length)

    # load data
    ts = extract.load_data(filePath, hypnogram=hypnogram)

    # get basic properties and components
    # [EEG-TSD, EOG-TSD,], [spindle_env]
    spindle_env = extract.basic_properties(ts, hypnogram,
                                           s_rate=s_rate,
                                           win_length=win_length,
                                           eeg_freq=eeg_freq,
                                           spindle_freq=spindle_freq,
                                           show_properties=False)

    eeg_energy, spindle_energy = extract.feature_extraction(ts, spindle_env,
                                                            start=start,
                                                            end=end,
                                                            s_rate=s_rate,
                                                            win_length=win_length,
                                                            eeg_freq=eeg_freq)

    # _____________________________________________________________________________________________________________

    with open('../testrun2.txt', 'a') as fp:
        tools.data2file(fp, eeg_energy,
                        hypno_array=hypno_array,
                        win_length=win_length,
                        s_rate=s_rate)

    '''
    
    
    
    
    def collect_data(fp):
        collection = []
        hypnocnt = 0
        startcnt = 0;
        endcnt = 3000;
        while True:
            ts = file.get_eeg(startcnt, endcnt)
            if len(ts) == 0 or len(ts) < 3000:
                print endcnt
                return collection
            if hypnocnt > len(hypno_array):
                print 'Nicht genug Hypnowerte!'
                exit(0)
            ts = tools.butter_bandpass_filter(ts, 0.15, 100)
            fft_ticks, powerfun = powerfunction(ts, 100)
            power_mat = tools.array_merge(fft_ticks, powerfun)
            delta, theta, alpha, beta, gamma = tools.band_slicer(power_mat)
            appender = [sum(delta), sum(theta), sum(alpha), sum(beta), sum(gamma), hypno_array[hypnocnt]]
            
            # collection.append(appender)
            hypnocnt = hypnocnt + 1
            startcnt = endcnt
            endcnt = endcnt + 3000


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



    #### do pre processing:
    # filtering
    filt = amri.amri_sig_filtfft(ts, fs=100, lowcut=12, highcut=14, trans=0.015, revfilt=True)
    # -- Normalize signal
    # filt = filt/stat.stdev(filt)
    # calculate threshold
    
    # calculate min-max distance and signal energy
    # energy = tools.energy_in_window(filt,100,window=30)
    # dist = tools.min_max_dist(filt,100,window = 0.5)


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

with open('../data/testrun.txt', 'a') as fp:
    collection = collect_data(fp)
    '''
