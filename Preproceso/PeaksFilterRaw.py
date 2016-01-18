"""
.. module:: PeaksFilterRaw

PeaksFilterRaw
*************

:Description: PeaksFilterRaw

 Filters the signals with a band-pass butterworth filter and saves the filtered signals
  and the identified peaks extracted from the filtered signal

:Authors: bejar
    

:Version: 

:Created on: 13/07/2015 8:37 

"""
import h5py
import numpy as np
from scipy.signal import butter, filtfilt

from Config.experiments import experiments, lexperiments
import argparse
__author__ = 'bejar'


def filter_data(experiment):
    """
    Filters and saves the raw signal in the datafile

    :param expname:
    :param iband:
    :param fband:
    :return:
    """


    f = h5py.File(experiment.dpath + experiment.name + '/' + experiment.name + '.hdf5', 'r+')
    print( experiment.dpath + experiment.name)

    # Window length in miliseconds from the peak identification
    wtime = experiment.peaks_id_params['wtime']
    sampling = experiment.sampling
    tw = int(2 * np.round(wtime * sampling / 2))
    iband = experiment.peaks_filter['lowpass']
    fband = experiment.peaks_filter['highpass']

    for df in experiment.datafiles:
        print(df)
        d = f[df + '/Raw']
        samp = f[df + '/Raw'].attrs['Sampling']
        data = d[()]
        freq = samp * 0.5
        b, a = butter(3, [iband / freq, fband / freq], btype='band')
        filtered = np.zeros(data.shape)
        for i in range(data.shape[1]):
            filtered[:, i] = filtfilt(b, a, data[:, i])
        d = f.require_dataset(df + '/RawFiltered', filtered.shape, dtype='f', data=filtered, compression='gzip')
        d[()] = filtered
        f[df + '/RawFiltered'].attrs['Low'] = iband
        f[df + '/RawFiltered'].attrs['high'] = fband
        for s in experiment.sensors:
            i = experiment.sensors.index(s)
            times = f[df + '/' + s + '/Time']
            rawpeaks = np.zeros((times.shape[0], tw))
            print(times.shape[0])
            for j in range(times.shape[0]):
                tstart = times[j] - np.floor(tw / 2)
                tstop = tstart + tw
                if tstart > 0 and tstop < filtered.shape[0]:
                    rawpeaks[j, :] = filtered[tstart:tstop, i]
                elif tstart < 0:
                    rawpeaks[j, :] = np.hstack((np.zeros(np.abs(tstart)), filtered[0:tstop, i]))
                else:
                    rawpeaks[j, :] = np.hstack((filtered[tstart:tstop, i], np.zeros(tstop - filtered.shape[0])))

            # Peak Data
            dfilter = f[df + '/' + s]
            dfilter.require_dataset('PeaksFilter', rawpeaks.shape, dtype='f', data=rawpeaks,
                                    compression='gzip')
            f[df + '/' + s + '/PeaksFilter'].attrs['lowpass'] = iband
            f[df + '/' + s + '/PeaksFilter'].attrs['highpass'] = fband
            f[df + '/' + s + '/PeaksFilter'].attrs['wtime'] = experiment.peaks_id_params['wtime']
            f[df+ '/' + s + '/PeaksFilter'].attrs['low'] = experiment.peaks_id_params['low']
            f[df + '/' + s + '/PeaksFilter'].attrs['high'] = experiment.peaks_id_params['high']
            f[df + '/' + s + '/PeaksFilter'].attrs['threshold'] = experiment.peaks_id_params['threshold']


    f.close()


# ---------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # 'e150514''e120503''e110616''e150707''e151126''e120511'
    lexperiments = ['e120511']

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()
    if args.exp:
        lexperiments = args.exp



    for exp in lexperiments:
        filter_data(experiments[exp])
