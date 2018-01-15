"""
 Separate the spectra of fractal component and oscillatory component from mixed time series
   amri_sig_fractal()

 Usage
   spec = amri_sig_fractal(sig,srate,...)

 Inputs
   sig   - a time-series vector. If sig is a matrix, then separate spectra for each column
   srate - sampling rate

 Outputs
   spec  - spectrum
           .freq = a vector of frequency points
           .srate= sample rate;
           .mixd = spectrum of mixed time series
           .frac = spectrum of fractal component
           .osci = spectrum of oscillatory component

 Keywords
   frange  - [fmin fmax](default [0 srate/4]), the output frequency range.
   detrend - 1 or 0 (default 1): 1 means detrending data before fft, otherwise 0
   filter  - 1 or 0 (default 1): 1 means filtering before downsampling to avoid aliasing.
   hset    - (default 1.1:0.05:1.9) an array containing scaling factors (> 1).

 See also
   amri_sig_genfrac
   amri_sig_plawfit

 Version
   0.10

 Reference
   -Wen H. and Liu Z. (2015) Separating Fractal and Oscillatory Components in
    the Power Spectrum of Neurophysiological Signals

 Translation to Pythen
    Martin Graf (martingraf@mail.de) University Osnabrueck (GER)

 History
 0.01  - HGWEN - 12/15/2013 - Use resample function instead of interp1
                            - upsample signal before extracting fractals
 0.02  - HGWEN - 01/15/2014 - add a new method 'nivcgsa'
                            - fit power-law line after resampling data in euqal space
                            - set nfft=2^nextpow2(2*Ndata);
 0.03  - HGWEN - 02/27/2014 - Change the name 'nivcgsa' to "IRASA".
 0.04  - HGWEN - 03/01/2014 - In IRASA, use median instead of min operator in the final step
 0.05  - HGWEN - 05/14/2014 - If sig is a matrix, separate spectra for each column
 0.06  - HGWEN - 08/20/2014 - Use multiple h values in CGSA
                            - remove the power-law fitting section, and add a new function "amri_sig_plawfit"
                            - Only return freq, srate, mixd, and frac.
 0.07  - HGWEN - 10/11/2014 - Add a keyword "upsample"
 0.08  - HGWEN - 10/19/2014 - Reorganized the code
 0.09  - HCWEN - 04/11/2015 - Added keywords 'hset' and 'filter', and removed the keyword 'upsample'
 0.10  - HGWEN - 09/26/2015 - Reorganized the structure.
 0.10p - MaGRF - 12/09/2017 - translating 0.10 version from Matlab to python 3.6


    # ARIMA
    # http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/
    # plot TS

"""

import math as m

import fractions
import numpy as np
import resampy
import scipy.interpolate as ip

import amri_sig_filtfft as af


# def irisa(data,sampling,subsets = 10,h = np.array(np.linspace[1.1,1.9,17]),filter = True):
def irisa(data, s_rate, subsets=15, h=np.asarray(np.linspace(1.1, 1.9, num=17)), filter=True):
    '''

    :param data: time series data  (TSD)
    :param s_rate: sampling rate of TSD in Hz
    :param subsets: number of subsets in IRASA
    :param h: upsampling parameters (should be an array with values > 1
    :param filter: flag - use anti aliasing filter
    :return:
    '''
    # convert data to type array
    data = np.asarray(data)
    # Length of time period
    n_period = len(data)
    # the highest power of 2 that does not exceed 90 % of n_period.
    n_data = int(pow(2, np.floor(m.log(n_period * 0.9, 2))))
    # time lag between the subsets
    subset_lag = np.floor((n_period - n_data) / (subsets - 1))
    # set nfft greater than ceil(h(end)) * Ndata, asure that do fft without truncating
    nfft = pow(2, af.nextpow2(m.ceil(max(h)) * n_data))
    # set output data length
    n_frac = int(nfft / 2 + 1)
    freq = (s_rate / 2) * np.linspace(0, 1, n_frac)

    # compute the spectrum of mixed data
    spec_mixed = np.zeros(n_frac)
    taper = np.hanning(n_data + 1)[0:n_data]  # periodic Hanning-window for fft

    # compute the mixed periodogram for all subsets
    for k in range(subsets):
        start = int(subset_lag * k)
        end = int(start + n_data)
        sample = (data[start:end] - np.mean(data[start:end])) * taper
        powerfun = 2 * np.fft.fft(sample, n=nfft) / min(nfft, n_data)
        powerfun[0] = powerfun[0] / 2
        spec_mixed = spec_mixed + pow(abs(powerfun[0:n_frac]), 2)

    spec_mixed = spec_mixed / subsets

    # filter the input signal to avoid aliasing when downsampling
    if filter == True:
        lowcut = 0
        highcut = s_rate / (2 * m.ceil(max(h)))
        data_filtered = af.amri_sig_filtfft(data, fs=s_rate, lowcut=lowcut, highcut=highcut)
    else:
        data_filtered = data

    # compute fractal component.
    spec_frac = np.zeros((len(h), n_frac))

    for j in range(len(h)):
        # compute the auto - power spectrum of xh
        Sh = np.zeros(n_frac)
        # compute the auto - power spectrum of X1h
        S1h = np.zeros(n_frac)
        # compute for every single subset
        for k in range(subsets):
            start = int(subset_lag * k)
            end = int(start + n_data)
            segment_ds = data_filtered[start:end]
            segment_us = data[start:end]
            segment_ds = myresample(segment_ds, 1. / h[j])
            segment_us = myresample(segment_us, h[j])
            taper_ds = np.hanning(len(segment_ds))
            taper_us = np.hanning(len(segment_us))
            powerfun_ds = 2 * np.fft.fft((segment_ds - np.mean(segment_ds)) * taper_ds, nfft) / min(nfft,
                                                                                                    len(segment_ds))
            powerfun_us = 2 * np.fft.fft((segment_us - np.mean(segment_us)) * taper_us, nfft) / min(nfft,
                                                                                                    len(segment_us))
            powerfun_ds[0] = powerfun[0] / 2
            powerfun_us[0] = powerfun[0] / 2
            # sum up periodograms in upsampling per h
            Sh = Sh + pow(abs(powerfun[0:n_frac]), 2)
            # sum up periodograms in downs_rate per h
            S1h = S1h + pow(abs(powerfun[0:n_frac]), 2)
        # take the mean of periodogram in upsampling per h
        Sh = Sh / subsets
        # take the mean of periodogram in downsampling per h
        S1h = S1h / subsets
        # take the square root of the product of up- and downsampling per h
        spec_frac[j][:] = np.sqrt(Sh * S1h)

    # pick the median element out of h periodograms  per frequency
    spec_frac = np.median(spec_frac, axis=0)

    return [spec_frac, spec_mixed, freq]


# subfunctions

def myresample(data, h, interp=False):
    # resample signal with upsample = numerator and downsample = denominator of h as fraction
    if interp == True:
        N = len(data)
        # initial scale
        x0 = np.linspace(0, 1, N)
        # resampling scale
        x1 = np.linspace(0, 1, np.round_(N * h))
        # create interpolation
        res_data = ip.interp1d(x0, data)(x1)

    elif interp == "cubic":
        # fit a model to the data
        env_model = ip.interp1d(np.linspace(1, len(data), num=len(data)), data, kind='cubic', bounds_error=False,
                                fill_value=0.0)
        # estimate model points for ts
        res_data = np.asarray(env_model(np.linspace(1, len(data), num=h * len(data))))


    else:
        up = fractions.Fraction(h).numerator
        down = fractions.Fraction(h).denominator
        res_data = resampy.resample(data, down, up)

    return np.asarray(res_data)
