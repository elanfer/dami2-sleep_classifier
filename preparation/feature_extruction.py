def get_features(data, sampling=1, windowing=False):
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
            fp.write(str(appender) + os.linesep)
            # collection.append(appender)
            hypnocnt = hypnocnt + 1
            startcnt = endcnt
            endcnt = endcnt + 3000
