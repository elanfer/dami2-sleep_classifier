"""

 amri_sig_filtfft() - lowpass highpass or bandpass/stop filtering using a pair of forward and inverse fourier transform.

 Usage
  ts_new = filtfft(ts, fs, {lowcut}, highcut, {revfilt}, {trans})

 Inputs
   ts:      a discrete time series vector
   fs:      sampling frequency of the time series {default = 1}
   lowcut:  lowcutoff frequency (in Hz)
   highcut: highcutoff frequency (in Hz)
   revfilt: 0:band-pass; 1:band-stop {default: 0}
   trans:   relative transition zone {default: 0.15}

 Output:
   ts_new:  the filtered time series vector

 See also:
  fft(),ifft()

 Version:
    1.06p  (Mathlab version 1.06)

 Examples:
   N/A

  DISCLAIMER AND CONDITIONS FOR USE:
     This software is distributed under the terms of the GNU General Public
     License v3, dated 2007/06/29 (see http://www.gnu.org/licenses/gpl.html).
     Use of this software is at the user's OWN RISK. Functionality is not
     guaranteed by creator nor modifier(s), if any. This software may be freely
     copied and distributed. The original header MUST stay part of the file and
     modifications MUST be reported in the 'MODIFICATION HISTORY'-section,
     including the modification date and the name of the modifier.

 CREATED:
     Oct. 1, 2009
     Zhongming Liu, PhD
     Advanced MRI, NINDS, NIH

 CONVERTED:
    Martin Graf, martingraf@mail.de

 MODIFICATION HISTORY
    1.00  - 10/01/2009 - ZMLIU - create the file based on eegfiltfft.m (EEGLAB)
    1.01  - 01/10/2010 - ZMLIU - using sin() to small the edge of cutoff
                               - use an nfft of a power of 2 
                               - use transition zones
    1.02  - 01/15/2010 - ZMLIU - fix a bug re highpass filter
    1.03  - 04/12/2010 - ZMLIU - rename to amri_sig_filtfft.m
    1.04  - 06/16/2010 - ZMLIU - remove dc and linear trend before filtering; 
                               - after filtering add the trend back
    1.05  - 07/22/2010 - ZMLIU - change the way of computing fres
            16/11/2011 - JAdZ  - v1.05 included in amri_eegfmri_toolbox v0.1
    1.06  - 07/15/2013 - ZMLIU - fix a bug against filter out of bounds 
    1.06p - 12/09/2017 - MAGRF - translating the 1.06 Matlab version to python 3.6.
    1.07p - 12/09/2017 - MAGRF - new input argument remove trend true/false, default value of trans to 0.01

"""

import math as m

import numpy as np
import scipy.signal as sc


def amri_sig_filtfft(ts, fs=1.0, lowcut=0.0, highcut=np.nan, trans=.15, revfilt=False, remove_trend=False):
    '''
        lowpass, highpass or bandpass/stop filtering using a pair of forward and inverse fourier transform.

    :param ts: a discrete time series vector
    :param fs: sampling frequency of the time series {default = 1}
    :param lowcut: lowcutoff frequency (in Hz)
    :param highcut: highcutoff frequency (in Hz)
    :param trans: 0:band-pass; 1:band-stop {default: 0}
    :param revfilt: relative transition zone {default: 0.15}
    :return: the filtered time series vector
    '''
    # number of time points
    npts = len(ts)
    # number of frequency points
    nfft = pow(2, nextpow2(npts))
    # even-sized frequency vector from 0 to nyguist frequency
    fv = fs / 2. * np.linspace(0., 1., int(nfft / 2 + 1))
    # frequency domain resolution
    fres = (fv[len(fv) - 1] - fv[0]) / (nfft / 2)

    # remove the linear trend
    if remove_trend == True:
        ts_old = ts
        ts = sc.detrend(ts_old, type='linear')
        trend = ts_old - ts
    else:
        trend = 0

    # desired frequency response
    filter = np.zeros(nfft) + 1

    # design frequency domain filter
    # ----------------------------------------------------------

    #
    # HIGHPASS FILTER:
    #
    #                lowcut
    #                  -----------
    #                 /
    #                /
    #               /
    #    -----------
    #         lowcut*(1-trans)

    if (~np.isnan(lowcut) and lowcut > 0) and (np.isnan(highcut) or highcut <= 0):

        idxl = int(round(lowcut / fres)) + 1
        idxlmt = int(round(lowcut * (1 - trans) / fres)) + 1
        filter[0:idxlmt] = 0
        filter[idxlmt:idxl] = 0.5 * (1 + np.sin(-m.pi / 2. + np.linspace(0, m.pi, idxl - idxlmt + 1)))
        filter_part = filter[0:idxl]
        filter[nfft - idxl - 1:nfft] = filter_part[::-1]



    #
    # LOWPASS FILTER:
    #
    #        highcut
    #       ----------
    #                 \
    #                  \
    #                   \
    #                    -----------
    #           highcut*(1+trans)

    elif (np.isnan(lowcut) or lowcut <= 0) and (~np.isnan(highcut) or highcut > 0):
        idxh = int(round(highcut / fres)) + 1
        idxhpt = int(round(highcut * (1 + trans) / fres)) + 1
        filter[idxh - 1:idxhpt] = 0.5 * (1. + np.sin(m.pi / 2. + np.linspace(0, m.pi, idxhpt - idxh + 1)))
        filter[idxhpt:int(nfft / 2)] = 0
        filter_part = filter[idxh - 1:int(nfft / 2)]
        filter[int(nfft / 2):int(nfft - idxh + 1)] = filter_part[::-1]


    elif (lowcut > 0 and highcut > 0 and highcut > lowcut):

        #
        # BANDPASS FILTER (revfilt = True)
        #
        #        lowcut                 highcut
        #                   -------
        #                  /       \    transition = (highcut - lowcut) / 2 * trans
        #                 /         \   center = (lowcut + highcut) / 2;
        #                /           \
        #         -------             -----------
        # lowcut - transition      highcut + transition

        if (revfilt == True):
            transition = (highcut - lowcut) / 2 * trans
            idxl = int(round(lowcut / fres)) + 1
            idxlmt = int(round((lowcut - transition) / fres)) + 1
            idxh = int(round(highcut / fres)) + 1
            idxhpt = int(round((highcut + transition) / fres)) + 1
            filter[0:idxlmt] = 0
            filter[idxlmt - 1:idxl] = 0.5 * (1 + np.sin(-m.pi / 2 + np.linspace(0, m.pi, idxl - idxlmt + 1)))
            filter[idxh - 1:idxhpt] = 0.5 * (1 + np.sin(m.pi / 2 + np.linspace(0, m.pi, idxhpt - idxh + 1)))
            filter[idxhpt - 1:int(nfft / 2)] = 0
            filter_part = filter[0:idxl]
            filter[nfft - idxl:nfft] = filter_part[::-1]
            filter_part = filter[idxh - 1:int(nfft / 2)]
            filter[int(nfft / 2):int(nfft - idxh + 1)] = filter_part[::-1]

        #
        # BANDSTOP FILTER (revfilt = False)
        #   lowcut - transition        highcut + transition
        #             -------             -----------
        #                    \           /
        #                     \         /   transition = (highcut - lowcut) / 2 * trans
        #                      \       /    center = (lowcut + highcut) / 2;
        #                       -------
        #             lowcut                 highcut

        else:
            transition = ((highcut - lowcut) / 2) * trans
            idxl = int(round(lowcut / fres)) + 1
            idxlmt = int(round((lowcut - transition) / fres)) + 1
            idxh = int(round(highcut / fres)) + 1
            idxhpt = int(round((highcut + transition) / fres)) + 1
            filter[idxlmt - 1:idxl] = 0.5 * (1 + np.sin(m.pi / 2 + np.linspace(0, m.pi, idxl - idxlmt + 1)))
            filter[idxh:idxhpt] = 0.5 * (1 + np.sin(-m.pi / 2 + np.linspace(0, m.pi, idxl - idxlmt + 1)))
            filter[idxl:idxh + 1] = 0
            filter_part = filter[idxlmt:idxhpt]
            filter[nfft - idxhpt:nfft - idxlmt + 1] = filter_part[::-1]

    else:
        print("amri_sig_filtfft(): error in lowcut and highcut setting")

    # fft
    X = np.fft.fft(ts, nfft)
    # ifft
    ts_new = np.real(np.fft.ifft(X * filter, nfft))
    # tranc
    ts_new = ts_new[0:npts]
    # add back the linear trend
    ts_new = ts_new + trend
    return ts_new


"""
Find 2^n that is equal to or greater than.
"""


def nextpow2(i):
    n = m.floor(np.log2(i)) + 1
    return n
