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
import argparse



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

__author__ = 'bejar'

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
        # 'e120503''e110616''e150707''e151126''e120511''e150514''e110906o''e160204'
        lexperiments = ['e150514']

    for expname in lexperiments:


        datainfo = experiments[expname]
        f = h5py.File(datainfo.dpath + datainfo.name + '/' + datainfo.name + '.hdf5', 'r')

        rslt = 50000
        step = 10000
        for dfile in range(0,len(datainfo.datafiles)):
            d = f[datainfo.datafiles[dfile] + '/' + 'Raw']
            length = int(d.shape[0]/float(step))
            lsignalsvar = []
            lsignalsmean = []
            for s in range(len(datainfo.sensors)):
                print dfile, datainfo.sensors[s]
                pvar = np.zeros(length)
                pmean = np.zeros(length)
                for pos in range(length):
                    pvar[pos] = np.std(d[pos*step:(pos*step)+rslt, s])
                #print p
                lsignalsvar.append((pvar,datainfo.sensors[s]))
                plt.subplots(figsize=(20, 10))
                plt.axis([0, length, 0, 0.2])
                plot(range(length), pvar)
                #plt.show()
                plt.title(datainfo.datafiles[dfile]+'-'+ datainfo.expnames[dfile]+ '-' + datainfo.sensors[s], fontsize=48)
                plt.savefig(datainfo.dpath + '/' + datainfo.name + '/Results/' + 'variance-' + datainfo.datafiles[dfile]+'-'+ datainfo.expnames[dfile] + '-' + datainfo.sensors[s]
                            + '.pdf', orientation='landscape', format='pdf')
                plt.close()
            plotSignals(lsignalsvar, 6, 2, 0.2, 0,
                        'variance-'+datainfo.datafiles[dfile]+'-'+datainfo.expnames[dfile]+'-'+str(rslt)+'-'+str(step),
                        datainfo.datafiles[dfile]+'-'+datainfo.expnames[dfile],
                        datainfo.dpath + '/' + datainfo.name + '/Results/', orientation='portrait', cstd=[0.05]*12)