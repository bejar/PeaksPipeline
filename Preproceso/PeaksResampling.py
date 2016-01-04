"""
.. module:: PeaksResampling

PeaksResampling
*************

:Description: PeaksResampling


 Resamples the peaks from the signals and saves the results in the HDF5 files as *Signal*/PeaksResample

 Hace el resampling de los datos originales o de los picos a los que se ha pasado un filtro que elimina frecuencias

:Authors: bejar
    

:Version: 

:Created on: 23/03/2015 11:41 

"""

__author__ = 'bejar'

from scipy.signal import resample
import h5py

from Config.experiments import experiments, lexperiments
from joblib import Parallel, delayed


def do_the_job(dfile, sensor, wtsel, resampfac, filter=False):
    """
    Applies a resampling of the data using Raw peaks
    The time window selected has to be larger than the length of the raw peaks

    wtsel = final length to keep from the resampled window in miliseconds
    resampfac = resampling factor (times to reduce the sampling)
    filter = use frequency filtered data

    :param expname:
    :return:
    """

    print(datainfo.dpath + datainfo.name, sensor)
    f = h5py.File(datainfo.dpath + datainfo.name + '/' + datainfo.name + '.hdf5', 'r')

    # Sampling of the dataset in Hz / resampling factor
    resampling = f[dfile + '/Raw'].attrs['Sampling'] / resampfac

    if dfile + '/' + sensor + '/' + 'PeaksFilter' in f or dfile + '/' + sensor + '/' + 'Peaks' in f:
        if filter:
            d = f[dfile + '/' + sensor + '/' + 'PeaksFilter']
        else:
            d = f[dfile + '/' + sensor + '/' + 'Peaks']

        data = d[()]
        f.close()

        # Number of samples in the peak
        wtlen = int(data.shape[1] / resampfac)
        wtlen_new = int(wtsel * resampling / 1000.0) # 1000 because the selection window is in miliseconds
        wtdisc = int((wtlen - wtlen_new)/2.0)
        presamp = resample(data, wtlen, axis=1, window=wtlen*2)

        # in case we have a odd number of points in the window
        if wtlen_new + (2*wtdisc) != wtlen:
            wtdisci = wtdisc + 1
        else:
            wtdisci = wtdisc
        return presamp[:, wtdisci:wtlen-wtdisc]
    return None

# ---------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # 'e110616''e120503''e150514''e150707'
    lexperiments = ['e151126']

    for expname in lexperiments:

        datainfo = experiments[expname]
        wtsel = datainfo.peaks_resampling['wtsel']
        resampfactor = datainfo.peaks_resampling['rsfactor']
        filtered = datainfo.peaks_resampling['filtered']  # Use the filtered peaks or not

        for dfile in datainfo.datafiles:
            print(dfile)
            # Paralelize PCA computation
            res = Parallel(n_jobs=-1)(delayed(do_the_job)(dfile, s, wtsel, resampfactor, filter=filtered) for s in datainfo.sensors)
            #print 'Parallelism ended'

            f = h5py.File(datainfo.dpath + datainfo.name + '/' + datainfo.name + '.hdf5', 'r+')
            for presamp, sensor in zip(res, datainfo.sensors):
                if presamp is not None:
                    print(dfile + '/' + sensor)
                    if dfile + '/' + sensor + '/' + 'PeaksResample' in f:
                        del f[dfile + '/' + sensor + '/' + 'PeaksResample']
                    d = f.require_dataset(dfile + '/' + sensor + '/' + 'PeaksResample', presamp.shape, dtype='f',
                                          data=presamp, compression='gzip')
                    d[()] = presamp
                    f[dfile + '/' + sensor + '/PeaksResample'].attrs['rsfactor'] = resampfactor
                    f[dfile + '/' + sensor + '/PeaksResample'].attrs['wtsel'] = wtsel
                    f[dfile + '/' + sensor + '/PeaksResample'].attrs['filtered'] = filtered

            f.close()
