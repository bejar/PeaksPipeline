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

from scipy.signal import resample, detrend
import h5py

from Config.experiments import experiments, lexperiments
from joblib import Parallel, delayed
import argparse
from itertools import product
import multiprocessing
from util.itertools import batchify

def do_the_job(dfile, sensor, wtsel, resampfac, rawfilter=False, dtrnd=False):
    """
    Applies a resampling of the data using Raw peaks
    The time window selected has to be larger than the length of the raw peaks

    wtsel = final length to keep from the resampled window in miliseconds
    resampfac = resampling factor (times to reduce the sampling)
    rawfilter = use frequency filtered raw data

    :param expname:
    :return:
    """
    print(dfile, sensor)
    f = datainfo.open_experiment_data(mode='r')

    # Sampling of the dataset in Hz / resampling factor
    resampling = f[dfile + '/Raw'].attrs['Sampling'] / resampfac

    if rawfilter:
        data = datainfo.get_peaks_filtered(f, dfile, sensor)
    else:
        data = datainfo.get_peaks(f, dfile, sensor)


    datainfo.close_experiment_data(f)

    if data is not None:
        if dtrnd:
            data = detrend(data)

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
        return presamp[:, wtdisci:wtlen-wtdisc], dfile, sensor
    else:
        return None, dfile, sensor

# ---------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")
    parser.add_argument('--detrend', help="Detrending of the signal after resampling", action='store_true', default=False)
    parser.add_argument('--extra', help="Procesa sensores extra del experimento", action='store_true', default=False)

    args = parser.parse_args()
    lexperiments = args.exp
    njobs = multiprocessing.cpu_count()

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511'e160317
        lexperiments = ['e130221c']
        args.detrend = False
        args.extra = False

    for expname in lexperiments:

        datainfo = experiments[expname]
        wtsel = datainfo.peaks_resampling['wtsel']
        resampfactor = datainfo.peaks_resampling['rsfactor']
        filtered = datainfo.peaks_resampling['filtered']  # Use the filtered peaks or not

        if not args.extra:
            lsensors = datainfo.sensors
        else:
            lsensors = datainfo.extrasensors


        batches = batchify([i for i in product(datainfo.datafiles, lsensors)], njobs)

        for batch in batches:
            # Paralelize PCA computation
            res =  Parallel(n_jobs=-1)(delayed(do_the_job)(dfile, sensor, wtsel, resampfactor, rawfilter=filtered, dtrnd=args.detrend) for dfile, sensor in batch)
            #print 'Parallelism ended'

            f = datainfo.open_experiment_data(mode='r+')
            for presamp, dfile, sensor in res:
                if presamp is not None:
                    print(dfile + '/' + sensor)
                    if not args.extra:
                        datainfo.save_peaks_resample(f, dfile, sensor, presamp)
                    else:
                        datainfo.save_peaks_resample(f, dfile, sensor, -presamp)

            datainfo.close_experiment_data(f)
