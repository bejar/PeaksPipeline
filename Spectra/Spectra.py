"""
.. module:: Spectra

Spectra
*************

:Description: Spectra

    

:Authors: bejar
    

:Version: 

:Created on: 11/06/2015 11:38 

"""
from __future__ import division

__author__ = 'bejar'


import h5py
from util.plots import show_signal, plotSignals
from util.distances import simetrized_kullback_leibler_divergence, square_frobenius_distance, renyi_half_divergence, \
    jensen_shannon_divergence, bhattacharyya_distance, hellinger_distance
import numpy as np
from sklearn.cluster import KMeans
from pylab import *
from scipy.signal import decimate, butter, filtfilt, freqs, lfilter

from Config.experiments import experiments
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from operator import itemgetter
import argparse

def filterSignal(data, iband, fband, freq):
    if iband == 1:
        print fband/freq
        b,a = butter(8, fband/freq, btype='low')
        flSignal = filtfilt(b, a, data)
    elif fband == 1:
        b, a = butter(8, fband/freq, btype='high')
        flSignal = filtfilt(b, a, data)
    else:
        print iband/freq, fband/freq
        b,a = butter(8, iband/freq, btype='high')
        temp = filtfilt(b, a, data)
        b,a = butter(8, fband/freq, btype='low')
        flSignal = filtfilt(b, a, temp)
    return flSignal

from scipy.signal import butter, lfilter, iirfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=8):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()
    lexperiments = args.exp

    if not args.batch:
       # 'e120503''e110616''e150707''e151126''e120511''e150514''e110906o'
        lexperiments = ['e150514']

    peakdata = {}
    for expname in lexperiments:

        datainfo = experiments[expname]
        f = h5py.File(datainfo.dpath + datainfo.name + '/' + datainfo.name + '.hdf5', 'r')


        for dfile in range(0,len(datainfo.datafiles)):
            d = f[datainfo.datafiles[dfile] + '/' + 'Raw']
            print d.shape, datainfo.sampling/2
            for s in range(len(datainfo.sensors)):
                print dfile, datainfo.sensors[s]
                rate = datainfo.sampling
                t = np.arange(0, 10, 1/rate)
                freq = rate * 0.5
                iband = 100.0
                fband = 400.0
                #b,a = butter(10, [iband/freq, fband/freq], btype='band')
                #b, a = iirfilter(2, 0.5, 1, 60, analog=True, ftype='cheby1', btype='low')
                # w, h = freqs(b, a, 1000)
                # print h[0:10]
                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                # ax.semilogx(w, 20 * np.log10(abs(h)))
                # ax.set_title('Chebyshev Type II bandpass frequency response')
                # ax.set_xlabel('Frequency [radians / second]')
                # ax.set_ylabel('Amplitude [dB]')
                # ax.axis((10, 1000, -100, 10))
                # ax.grid(which='both', axis='both')
                # plt.show()

                # # data =  d[0:60000, s].copy()
                # # x = lfilter(b, a, data, )
                # print x[0:100], data[0:100]
                # plt.subplots(figsize=(20, 10))
                # plot(range(1000), x[0:1000])
                # plot(range(1000), data[0:1000])
                # plt.show()
                # print x.shape
#                print x[0:100], d[0:100,s]
                p = np.abs(np.fft.rfft(d[0:6000000,s]))**2
                spec = np.linspace(0, (rate)/2, len(p))
                #p = decimate(p,20)
                plt.subplots(figsize=(20, 10))
                plot(spec[0:12000], p[0:12000])
                plt.show()
                # plt.title(datainfo.datafiles[dfile]+ '-' + datainfo.sensors[s], fontsize=48)
                # plt.savefig(datainfo.dpath + '/Results/' + datainfo.datafiles[dfile] + '-' + datainfo.sensors[s]
                #             + '-spectra.pdf', orientation='landscape', format='pdf')
                # plt.close()
