"""
.. module:: PeaksOutliers

PeaksOutliers
*************

:Description: PeaksOutliers

 Eliminates from the signal all the outliers and the too wavy signals

:Authors: bejar

:Version: 

:Created on: 14/07/2015 9:00

* 4/2/2016 - Adapting to changes in Experiment class

"""

__author__ = 'bejar'


from pylab import *
from sklearn.neighbors import NearestNeighbors
from Config.experiments import experiments
from joblib import Parallel, delayed
from util.plots import show_signal
import argparse


def is_wavy_signal(signal, thresh):
    """
    Detects if a signal has many cuts over the positive middle part of the signal

    :param signal:
    :return:
    """

    middle = np.max(signal)*0.3
    tmp = signal.copy()

    tmp[signal < middle] = 0

    count = 0
    for i in range(1, signal.shape[0]):
        if tmp[i-1] == 0 and tmp[i] != 0:
            count += 1
    return count > thresh


def do_the_job(dpath, dname, dfile, sensor, nn, nstd=6, wavy=5):
    """
    Identifies the outliers in the peaks

    :param dfile:
    :param sensor:
    :return:
    """

    # Detect outliers based on the distribution of the distances of the signals to the knn
    # Any signal that is farther from its neighbors that a number of standar deviations of the mean knn-distance is out
    print('Processing ', sensor, dfile)
    f = datainfo.open_experiment_data(mode='r')
    data = datainfo.get_peaks_resample(f, dfile, sensor)
    neigh = NearestNeighbors(n_neighbors=nn)
    neigh.fit(data)

    vdist = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        vdist[i] = np.sum(neigh.kneighbors(data[i].reshape(1, -1), return_distance=True)[0][0][1:])/(nn-1)
    dmean = np.mean(vdist)
    dstd = np.std(vdist)
    nout = 0
    lout = []
    for i in range(data.shape[0]):
        if vdist[i] > dmean + (nstd*dstd):
            nout += 1
            lout.append(i)
            # print('outlier')
            # show_signal(data[i])
        elif wavy is not None and is_wavy_signal(data[i], wavy):
            nout += 1
            lout.append(i)
            # print('wavy')
            # show_signal(data[i])
        # else:
        #     show_signal(data[i])

    datainfo.close_experiment_data(f)
    return dfile, lout


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()
    lexperiments = args.exp

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511'
        lexperiments = ['e120511']



    for expname in lexperiments:
        datainfo = experiments[expname]

        for sensor in datainfo.sensors:
            print(sensor)

            lout = Parallel(n_jobs=-1)(delayed(do_the_job)(datainfo.dpath, datainfo.name, dfiles, sensor, 16, nstd=6, wavy=4) for dfiles in datainfo.datafiles)

            # lout = []
            #
            # for dfiles in [datainfo.datafiles[0]]:
            #     lout.append(do_the_job(datainfo.dpath, datainfo.name, dfiles, s, 16, nstd=6, wavy=6))


            f = datainfo.open_experiment_data(mode='r+')
            for dfile, out in lout:

                times = datainfo.get_peaks_time(f, dfile, sensor)
                if times is not None:
                    ntimes = np.zeros(times.shape[0]-len(out))
                    npeaks = 0
                    for i in range(times.shape[0]):
                        if i not in out:
                            ntimes[npeaks] = i
                            npeaks += 1
                    datainfo.save_peaks_time_clean(f, dfile, sensor, ntimes)
                    print(times.shape, ntimes.shape, len(out))
            datainfo.close_experiment_data(f)
