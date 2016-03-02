"""
.. module:: PeaksClustering

PeaksClustering
*************

:Description: PeaksClustering

    Clusters the Peaks from an experiment all the files together

    Hace un clustering de los picos de cada sensor usando el numero de clusters indicado en la
    definicion del experimento y el conjunto de colores para el histograma de la secuencia del experimento

:Authors: bejar
    

:Version: 

:Created on: 26/03/2015 8:10 

"""

from collections import Counter
from operator import itemgetter

import h5py
import matplotlib.pyplot as plt
from pylab import *
from sklearn.cluster import KMeans

from Config.experiments import experiments
from sklearn.metrics import pairwise_distances_argmin_min
import argparse

__author__ = 'bejar'


def compute_data_labels(dfilec, dfile, sensor):
    """
    Computes the labels of the data using the centroids of the cluster in the file
    :param dfile:
    :param sensor:
    :return:
    """
    f = h5py.File(datainfo.dpath + '/' + datainfo.name + '/' + datainfo.name  + '.hdf5', 'r')

    d = f[dfilec + '/' + sensor + '/Clustering/' + 'Centers']
    centers = d[()]
    d = f[dfile + '/' + sensor + '/' + 'PeaksResamplePCA']
    data = d[()]
    labels, _ = pairwise_distances_argmin_min(data, centers)
    f.close()
    return labels


def show_vsignals(signal, centroid, title=''):
    """
    Plots a list of signals
    :param signal:
    :return:
    """
    matplotlib.rcParams.update({'font.size': 12})
    fig = plt.figure()
    fig.set_figwidth(30)
    fig.set_figheight(40)
    fig.suptitle(title, fontsize=48)
    minaxis = np.min(signal)
    maxaxis = np.max(signal)
    num = signal.shape[1]

    npeaks = signal.shape[0]

    nrows = (npeaks+1) / 5
    if (npeaks+1) % 5 != 0:
        nrows += 1


    t = arange(0.0, num, 1)
    for i in range(signal.shape[0]):
        sp1 = fig.add_subplot(5, nrows, i+1)
        sp1.axis([0, num, minaxis, maxaxis])
        sp1.plot(t, signal[i,:])

    sp1 = fig.add_subplot(5, nrows, npeaks+1)
    sp1.axis([0, num, minaxis, maxaxis])
    sp1.plot(t, centroid, color='r', linewidth=8.0)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")
    parser.add_argument('--pca', help="Show PCA transformed peaks", action='store_true', default=False)

    args = parser.parse_args()
    lexperiments = args.exp
    npeaks = 25

    if not args.batch:
        # 'e120503''e110616''e150707''e151126''e120511''e150514''e110906o'
        lexperiments = ['e150514']
        args.pca = False

    for expname in lexperiments:
        datainfo = experiments[expname]
        colors = datainfo.colors

        f = datainfo.open_experiment_data(mode='r')

        for dfile in datainfo.datafiles:
            print(dfile)

            ldata = []
            for sensor in datainfo.sensors:
                print(sensor)

                clpeaks = compute_data_labels(datainfo.datafiles[0], dfile, sensor)

                if args.pca:
                    dataf = datainfo.get_peaks_resample_PCA(f, dfile, sensor)
                else:
                    dataf = datainfo.get_peaks_resample(f, dfile, sensor)

                cnt = Counter(list(clpeaks))
                clustering = datainfo.get_clustering(f, datainfo.datafiles[0], sensor)
                for i in np.unique(clpeaks):
                    print len(dataf[clpeaks == i, :])
                    dpeaks = dataf[clpeaks == i, :]
                    show_vsignals(dpeaks[0:npeaks, :], clustering[i], dfile+'-'+sensor)

        datainfo.close_experiment_data(f)

