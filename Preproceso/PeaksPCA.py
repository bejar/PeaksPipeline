"""
.. module:: PeaksPCA

PeaksPCA
*************

:Description: PeaksPCA

    Performs a PCA of the resampled peaks and reconstructs them with only a number of them.
    After, it removes the mean a subwindow of the initial and final values of the signal
    Eventually saves the peaks in *Signal*/PeaksResamplePCA

    Uses joblib for paralelization

:Authors: bejar
    

:Version: 

:Created on: 26/03/2015 7:52 

"""


import h5py
import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

from Config.experiments import experiments

__author__ = 'bejar'


def do_the_job(dfile, sensor, components, lind, pcap=True):
    """
    Transforms the data reconstructing the peaks using some components of the PCA
    and uses the mean of the baseline points to move the peak

    :param pcap: Perform or not PCA
    :param dfile: datafile
    :param sensor: sensor
    :param components: Components selected from the PCA
    :param lind: Points to use to move the peak
    :return:
    """
    print(datainfo.dpath + datainfo.name, sensor)
    f = h5py.File(datainfo.dpath + datainfo.name + '/' + datainfo.name + '.hdf5', 'r')

    d = f[dfile + '/' + sensor + '/' + 'PeaksResample']
    data = d[()]

    if pcap:
        pca = PCA(n_components=data.shape[1])
        res = pca.fit_transform(data)

        print('VEX=', np.sum(pca.explained_variance_ratio_[0:components]))

        res[:, components:] = 0
        trans = pca.inverse_transform(res)
    else:
        trans = data

    # Substract the basal
    for row in range(trans.shape[0]):
        vals = trans[row, lind]
        basal = np.mean(vals)
        trans[row] -= basal

    f.close()
    return trans


# ---------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    lexperiments = ['e150514']

    for expname in lexperiments:

        datainfo = experiments[expname]
        fpca = datainfo.peaks_smooth['pcasmooth']
        components = datainfo.peaks_smooth['components']
        baseline = datainfo.peaks_smooth['wbaseline']
        lind = range(baseline)

        for dfile in datainfo.datafiles:
            print(dfile)
            # Paralelize PCA computation
            res = Parallel(n_jobs=-1)(
                delayed(do_the_job)(dfile, s, components, lind, pcap=fpca) for s in datainfo.sensors)
            # print 'Parallelism ended'
            # Save all the data
            f = h5py.File(datainfo.dpath + datainfo.name + '/' + datainfo.name + '.hdf5', 'r+')
            for trans, sensor in zip(res, datainfo.sensors):
                print(dfile + '/' + sensor + '/' + 'PeaksResamplePCA')
                if dfile + '/' + sensor + '/' + 'PeaksResamplePCA' in f:
                    del f[dfile + '/' + sensor + '/' + 'PeaksResamplePCA']
                d = f.require_dataset(dfile + '/' + sensor + '/' + 'PeaksResamplePCA', trans.shape, dtype='f',
                                      data=trans, compression='gzip')
                if fpca:
                    f[dfile + '/' + sensor + '/PeaksResamplePCA'].attrs['Components'] = components
                else:
                    f[dfile + '/' + sensor + '/PeaksResamplePCA'].attrs['Components'] = 0

                f[dfile + '/' + sensor + '/PeaksResamplePCA'].attrs['baseline'] = baseline

                d[()] = trans

            f.close()
