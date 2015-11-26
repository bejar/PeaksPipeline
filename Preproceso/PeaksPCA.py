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


def do_the_job(dfile, sensor, components, lind, pcap=True, recenter=True, wtsel=None):
    """
    Transforms the data reconstructing the peaks using some components of the PCA
    and uses the mean of the baseline points to move the peak

    :param pcap: Perform or not PCA
    :param dfile: datafile
    :param sensor: sensor
    :param components: Components selected from the PCA
    :param lind: Points to use to move the peak
    :param recenter: recenters the peak so it is in the center of the window
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

    # If recenter, find the new center of the peak and crop the data to wtsel milliseconds
    if recenter:
        # Original window size in milliseconds
        wtsel_orig = f[dfile + '/' + sensor + '/PeaksResample'].attrs['wtsel']
        # current window midpoint
        midpoint = int(trans.shape[1]/2.0)
        # New window size
        wtlen = int(trans.shape[1]*(wtsel/wtsel_orig))
        wtdisc = int((trans.shape[1] - wtlen)/2.0)
        # in case we have a odd number of points in the window
        if wtlen + (2*wtdisc) != wtlen:
            wtdisci = wtdisc + 1
        else:
            wtdisci = wtdisc

        new_trans = np.zeros((trans.shape[0], wtlen))
        for pk in range(trans.shape[0]):
            # find current maximum around the midpoint of the current window
            # Fixed to 10 points around the center
            center = np.argmax(trans[pk, midpoint-10:midpoint+10])
            new_trans[pk] = trans[pk,wtdisci:wtlen-wtdisc]

        trans = new_trans

    # Substract the basal
    for row in range(trans.shape[0]):
        vals = trans[row, lind]
        basal = np.mean(vals)
        trans[row] -= basal

    f.close()
    return trans


# ---------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # 'e110616''e120503'
    lexperiments = ['e150514']

    for expname in lexperiments:

        datainfo = experiments[expname]
        fpca = datainfo.peaks_smooth['pcasmooth']
        components = datainfo.peaks_smooth['components']
        baseline = datainfo.peaks_smooth['wbaseline']
        if 'recenter' in datainfo.peaks_smooth:
            # If recenter is true a subwindow of the data has to be indicated to be able to re-crop the signal
            recenter = datainfo.peaks_smooth['recenter']
            wtsel = datainfo.peaks_smooth['wtsel']
        else:
            recenter = False
            wtsel = None
        lind = range(baseline)

        for dfile in datainfo.datafiles:
            print(dfile)
            # Paralelize PCA computation
            res = Parallel(n_jobs=-1)(
                delayed(do_the_job)(dfile, s, components, lind, pcap=fpca, recenter=recenter, wtsel=wtsel) for s in datainfo.sensors)
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
                if recenter:
                    f[dfile + '/' + sensor + '/PeaksResamplePCA'].attrs['baseline'] = recenter
                    f[dfile + '/' + sensor + '/PeaksResamplePCA'].attrs['wtsel'] = wtsel

                d[()] = trans

            f.close()
