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
from scipy.stats import pearsonr

__author__ = 'bejar'


def compute_reference(datainfo, dfile, rsensor):
    """
    Computes the correlations for a file to use as reference

    :return:
    """
    f = datainfo.open_experiment_data(mode='r')

    datainfo.close_experiment_data(f)
    clpeaks = datainfo.compute_peaks_labels(f, dfile, rsensor)
    pktimes = datainfo.get_peaks_time(f, dfile, rsensor)

    IPFs, IPFp = datainfo.get_IPF_time_windows(f, dfile, pktimes, 1000)
    swindows = datainfo.get_sensors_time_windows(f, dfile, pktimes, 1000)
    for i in np.unique(clpeaks):
        dIPFs = IPFs[clpeaks == i, :]
        dIPFp = IPFp[clpeaks == i, :]
        clsensors = []
        for j, sensor in enumerate(datainfo.sensors):
            clsensors.append(swindows[j][clpeaks == i, :])

        lcorrs = np.zeros(len(datainfo.sensors))
        lcorrp = np.zeros(len(datainfo.sensors))
        for k, cl in enumerate(clsensors):
            for j in range(dIPFs.shape[0]):
                lcorrs[k] += pearsonr(dIPFs[j], cl[j])[0]
                lcorrp[k] += pearsonr(dIPFp[j], cl[j])[0]
        lcorrs /= dIPFs.shape[0]
        lcorrp /= dIPFs.shape[0]
    return lcorrs, lcorrp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()
    lexperiments = args.exp
    nsensor = 6

    if not args.batch:
        # 'e120503''e110616''e150707''e151126''e120511''e150514''e110906o'
        lexperiments = ['e110906o']
        args.pca = False

    for expname in lexperiments:
        datainfo = experiments[expname]
        colors = datainfo.colors
        rsensor = datainfo.sensors[nsensor]

        rlcorrs, rlcorrp = compute_reference(datainfo, datainfo.datafiles[0], rsensor)

        f = datainfo.open_experiment_data(mode='r')

        for dfile, ename in zip(datainfo.datafiles, datainfo.expnames):
            print(dfile)

            clpeaks = datainfo.compute_peaks_labels(f, dfile, rsensor)
            pktimes = datainfo.get_peaks_time(f, dfile, rsensor)
            clustering = datainfo.get_peaks_clustering_centroids(f, datainfo.datafiles[0], rsensor, datainfo.clusters[0])

            IPFs, IPFp = datainfo.get_IPF_time_windows(f, dfile, pktimes, 1000)
            swindows = datainfo.get_sensors_time_windows(f, dfile, pktimes, 1000)

            #lclcorr = []

            matplotlib.rcParams.update({'font.size': 10})
            fig = plt.figure()
            fig.set_figwidth(20)
            fig.set_figheight(10)
            fig.suptitle('%s-%s-%s' % (dfile, ename, rsensor), fontsize=20)
            nplots = len(np.unique(clpeaks))
            for i in np.unique(clpeaks):
                dIPFs = IPFs[clpeaks == i, :]
                dIPFp = IPFp[clpeaks == i, :]
                clsensors = []
                for j, sensor in enumerate(datainfo.sensors):
                    clsensors.append(swindows[j][clpeaks == i, :])

                lcorrs = np.zeros(len(datainfo.sensors))
                lcorrp = np.zeros(len(datainfo.sensors))
                for k, cl in enumerate(clsensors):
                    for j in range(dIPFs.shape[0]):
                        lcorrs[k] += pearsonr(dIPFs[j], cl[j])[0]
                        lcorrp[k] += pearsonr(dIPFp[j], cl[j])[0]
                lcorrs /= dIPFs.shape[0]
                lcorrp /= dIPFs.shape[0]

                dim = (len(datainfo.sensors) / 2) +1
                if (len(datainfo.sensors) % 2) != 0:
                    dim += 1

                mpls = np.zeros((dim, 2))
                mplp = np.zeros((dim, 2))

                rmpls = np.zeros((dim, 2))
                rmplp = np.zeros((dim, 2))

                for j in range(len(datainfo.sensors)):
                    if (j % 2) == 0:
                        mpls[j/2, 0] = lcorrs[j]
                        mplp[j/2, 0] = lcorrp[j]
                        rmpls[j/2, 0] = lcorrs[j] - rlcorrs[j]
                        rmplp[j/2, 0] = lcorrp[j] - lcorrp[j]

                    else:
                        mpls[j/2, 1] = lcorrs[j]
                        mplp[j/2, 1] = lcorrp[j]
                        rmpls[j/2, 1] = lcorrs[j] - rlcorrs[j]
                        rmplp[j/2, 1] = lcorrp[j] - lcorrp[j]

                mpls[dim-1, 0] = -1
                mplp[dim-1, 0] = -1
                mpls[dim-1, 1] = 1
                mplp[dim-1, 1] = 1
                rmpls[dim-1, 0] = -1
                rmplp[dim-1, 0] = -1
                rmpls[dim-1, 1] = 1
                rmplp[dim-1, 1] = 1

                sp1 = fig.add_subplot(3, nplots, i + 1)
                sp1.set_yticks(range(dim))
                sp1.set_xticks([0, 1])
                sp1.set_yticklabels(['L4c', 'L5r', 'L5c', 'L6r', 'L6c', 'L7r', 'R'])
                sp1.set_xticklabels(['I', 'D'])
                sp1.imshow(mpls, interpolation='none', cmap=plt.cm.seismic)

                sp1 = fig.add_subplot(3, nplots,  nplots + i + 1)
                sp1.set_yticks(range(dim))
                sp1.set_xticks([0, 1])
                sp1.set_yticklabels(['L4c', 'L5r', 'L5c', 'L6r', 'L6c', 'L7r', 'R'])
                sp1.set_xticklabels(['I', 'D'])
                sp1.imshow(mplp, interpolation='none', cmap=plt.cm.seismic)#lanczos

                sp1 = fig.add_subplot(3, nplots, (nplots*2) + i + 1)
                t = arange(0.0, clustering.shape[1], 1)
                sp1.axis([0, clustering.shape[1] , np.min(clustering), np.max(clustering)])
                sp1.axhline(linewidth=1, color='b', y=0)
                sp1.set_xticklabels([])
                sp1.plot(t, clustering[i], color='r', linewidth=3.0)

            plt.tight_layout(pad=25, w_pad=25, h_pad=25)
            plt.subplots_adjust(hspace = 0.1, wspace=2, top=0.95, bottom=0.05, left=0.05, right=0.95)
            #plt.show()

            fig.savefig(datainfo.dpath + datainfo.name + '/Results/IPFAll-' + datainfo.name  + '-' + dfile + '-' + ename
                        + '-' + rsensor + '.pdf',  format='pdf')
            plt.close()
        datainfo.close_experiment_data(f)

