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
from scipy.signal import detrend

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


def show_vsignals(lsignals, mnIPF, mxIPF, mncl, mxcl, dfile, sensor):
    """
    Plots a list of signals
    :param signal:
    :return:
    """
    matplotlib.rcParams.update({'font.size': 12})
    fig = plt.figure()
    fig.set_figwidth(30)
    fig.set_figheight(40)
    fig.suptitle(dfile + '-' + sensor, fontsize=48)

    for i, signals in enumerate(lsignals):
        IPFs = signals[0]
        IPFp = signals[1]
        centroid = signals[2]

        minaxis = -1 # mnIPF
        maxaxis = 1 # mxIPF
        num = IPFs[0].shape[0]
        t = arange(0.0, num, 1)

        sp1 = fig.add_subplot(len(lsignals), 3, (3 * i) + 1)
        sp1.axis([0, num, minaxis, maxaxis])
        sp1.plot(t, IPFs[0], color='r', linewidth=3.0)
        sp1.plot(t, IPFs[0] + IPFs[3], color='g', linewidth=1.0)
        sp1.plot(t, IPFs[0] - IPFs[3], color='g', linewidth=1.0)
        sp1.plot(t, IPFs[1], color='b', linewidth=2.0)
        sp1.plot(t, IPFs[2], color='b', linewidth=2.0)

        minaxis = -1 # mnIPF
        maxaxis = 1 # mxIPF
        num = IPFp[0].shape[0]
        t = arange(0.0, num, 1)

        sp1 = fig.add_subplot( len(lsignals), 3,(3 * i) + 2)
        sp1.axis([0, num, minaxis, maxaxis])
        sp1.plot(t, IPFp[0], color='r', linewidth=3.0)
        sp1.plot(t, IPFp[0] + IPFp[3], color='g', linewidth=1.0)
        sp1.plot(t, IPFp[0] - IPFp[3], color='g', linewidth=1.0)
        sp1.plot(t, IPFp[1], color='b', linewidth=2.0)
        sp1.plot(t, IPFp[2], color='b', linewidth=2.0)

        num = centroid.shape[0]
        t = arange(0.0, num, 1)

        sp1 = fig.add_subplot( len(lsignals), 3,(3 * i) + 3)
        sp1.axis([0, num, mncl, mxcl])
        plt.axhline(linewidth=1, color='b', y=0)
        sp1.plot(t, centroid, color='r', linewidth=3.0)

        #plt.title(dfile + ' - ' + sensor)

    fig.savefig(datainfo.dpath + datainfo.name + '/Results/IPFclust-' + datainfo.name  + '-' + dfile + '-' + sensor + '.pdf',  format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    #plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()
    lexperiments = args.exp

    if not args.batch:
        # 'e120503''e110616''e150707''e151126''e120511''e150514''e110906o'
        lexperiments = ['e110906o']
        args.pca = False

    for expname in lexperiments:
        datainfo = experiments[expname]
        colors = datainfo.colors

        f = datainfo.open_experiment_data(mode='r')

        for dfile, ename in zip(datainfo.datafiles, datainfo.expnames):
            print(dfile)

            ldata = []
            for sensor in datainfo.sensors:
                print(sensor)

                clpeaks = compute_data_labels(datainfo.datafiles[0], dfile, sensor)
                pktimes = datainfo.get_peaks_time(f, dfile, sensor)

                IPFs, IPFp = datainfo.get_IPF_time_windows(f, dfile, pktimes, 1000)
                clustering = datainfo.get_peaks_clustering_centroids(f, datainfo.datafiles[0], sensor, datainfo.clusters[0])
                mncl = np.min(clustering)
                mxcl = np.max(clustering)

                lsignals = []
                mnIPF = 1
                mxIPF = -1
                for i in np.unique(clpeaks):

                    dIPFs = IPFs[clpeaks == i, :]
                    dIPFp = IPFp[clpeaks == i, :]

                    avIPFs = dIPFs.mean(0)
                    mnIPFs = dIPFs.min(0)
                    mxIPFs = dIPFs.max(0)
                    stdIPFs = dIPFs.std(0)
                    avIPFp = dIPFp.mean(0)
                    mnIPFp = dIPFp.min(0)
                    mxIPFp = dIPFp.max(0)
                    stdIPFp = dIPFp.std(0)

                    mnIPF = np.min([mnIPF, mnIPFs.min(0), mnIPFp.min(0)])

                    mxIPF = np.max([mxIPF, mxIPFs.max(0), mxIPFp.max(0)])


                    lsignals.append([[avIPFs, mnIPFs, mxIPFs, stdIPFs], [avIPFp, mnIPFp, mxIPFp, stdIPFp], clustering[i]])

                show_vsignals(lsignals, mnIPF, mxIPF, mncl, mxcl, dfile+'-'+ename, sensor)

        datainfo.close_experiment_data(f)

