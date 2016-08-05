"""
.. module:: PeaksClusteringViewExamples

PeaksClustering
*************

:Description: PeaksClusteringViewExamples

 Visualiza picos de los clusters

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
from util.plots import show_signal
import seaborn as sn

__author__ = 'bejar'


def show_vsignals(signal, centroid, mnvals, mxvals, stdvals, title=''):
    """
    Plots a list of signals
    :param signal:
    :return:
    """
    matplotlib.rcParams.update({'font.size': 12})
    fig = plt.figure()
    fig.set_figwidth(30)
    fig.set_figheight(40)
    fig.suptitle(title, fontsize=20)
    minaxis = np.min(signal)
    maxaxis = np.max(signal)
    num = signal.shape[1]

    npeaks = signal.shape[0]

    nrows = (npeaks+1) / 5
    if (npeaks+1) % 5 != 0:
        nrows += 1


    t = arange(0.0, num, 1)
    for i in range(signal.shape[0]):
        imin = np.min(signal[i,0:num/2])
        fmin = np.min(signal[i,num/2:3*num/4])

        sp1 = fig.add_subplot(5, nrows, i+1)
        sp1.axis([0, num, minaxis, maxaxis])
        sp1.plot(t, signal[i,:], color='g', linewidth=1.0)
        if imin > fmin:
            plt.title('NP')
        else:
            plt.title('P')



        plt.axhline(linewidth=1, color='r', y=0)
    sp1 = fig.add_subplot(5, nrows, npeaks+1)
    minaxis = np.min(mnvals)
    maxaxis = np.max(mxvals)

    sp1.axis([0, num, minaxis, maxaxis])
    sp1.plot(t, centroid, color='r', linewidth=2.0)
    plt.axhline(linewidth=1, color='b', y=0)
    # sp1.plot(t, centroid + stdvals, color='g', linewidth=3.0)
    # sp1.plot(t, centroid - stdvals, color='g', linewidth=3.0)
    # sp1.plot(t, mxvals, color='b', linewidth=3.0)
    # sp1.plot(t, mnvals, color='b', linewidth=3.0)

    plt.show()

def plot_mins_distribution():
    """

    :return:
    """
    # fig = plt.figure()
    # fig.set_figwidth(30)
    # fig.set_figheight(40)
    # fig.suptitle(sensor + ' ' + str(i), fontsize=20)
    # minaxis = min(clustering[i])
    # maxaxis = max(clustering[i])
    #
    # sp1 = fig.add_subplot(121)
    # sp1.axis([0, num, minaxis, maxaxis])
    # t = arange(0.0, num, 1)
    # sp1.plot(t, clustering[i])
    #
    # plt.axhline(linewidth=4, color='r', y=0)
    #
    # sp1 = fig.add_subplot(122)
    # sn.distplot(limin, color='r')
    # sn.distplot(lfmin, color='b')
    # plt.show()
    # plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")
    parser.add_argument('--pca', help="Show PCA transformed peaks", action='store_true', default=False)
    parser.add_argument('--globalclust', help="Use a global computed clustering", action='store_true', default=False)

    args = parser.parse_args()
    lexperiments = args.exp


    if not args.batch:
        # 'e120503''e110616''e150707''e151126''e120511''e150514''e110906o'
        lexperiments = ['e150514']
        args.pca = True
        args.globalclust = False

    for expname in lexperiments:
        datainfo = experiments[expname]
        colors = datainfo.colors

        f = datainfo.open_experiment_data(mode='r')

        sensor = 'L5ci'
        fig = plt.figure()
        fig.set_figwidth(30)
        fig.set_figheight(18)
        fig.suptitle(expname + ' ' + sensor, fontsize=40)

        ncol = len(datainfo.expnames)
        nclusters = nfil = datainfo.clusters[datainfo.sensors.index(sensor)]


        columna = 0
        for dfile, ename in zip(datainfo.datafiles, datainfo.expnames):

            ldata = []

            # for sensor in datainfo.sensors:

            clpeaks = datainfo.compute_peaks_labels(f, dfile, sensor, nclusters, globalc=args.globalclust)

            datafPCA = datainfo.get_peaks_resample_PCA(f, dfile, sensor)
            if args.pca:
                dataf = datafPCA
            else:
                dataf = datainfo.get_peaks_resample(f, dfile, sensor)

            cnt = Counter(list(clpeaks))
            clustering = datainfo.get_peaks_clustering_centroids(f, datainfo.datafiles[0], sensor, nclusters)
            num = dataf.shape[1]
            for i in np.unique(clpeaks):
                pknp = 0
                pkp = 0

                # print len(dataf[clpeaks == i, :])
                limin = []
                lfmin = []
                dpeaks = dataf[clpeaks == i, :]
                for pk in dpeaks:
                    limin.append(np.min(pk[0:num/2]))
                    #lfmin.append(np.min(pk[num/2:]))
                iminstd = np.std(limin)
                iminmean = np.mean(limin)



                for pk in dpeaks:
                    fmin = np.min(pk[num/2:])
                    if fmin >= (iminmean - iminstd):
                        pkp += 1
                    else:
                        pknp += 1
                print dfile, ',',  ename, ',', sensor, ',', i+1, ',', (1.0* pkp)/len(dataf[clpeaks == i, :]),',',  (1.0* pknp)/len(dataf[clpeaks == i, :])

                sp1 = fig.add_subplot(nfil, ncol, (i * ncol) + (columna+1))

                x = np.array(list("+-"))
                y1 = np.array([(1.0* pkp)/len(dataf[clpeaks == i, :]), (1.0* pknp)/len(dataf[clpeaks == i, :])])
                fg = sn.barplot(x, y1, palette="BuGn_d",)
                fg.axes.set_ylim(0,1)
                fg.axes.set_yticks([])
                if ((i +1) % nclusters) == 0:
                    fg.axes.set_xlabel(ename)
                if columna == 0:
                    fg.axes.set_ylabel('C'+str(i+1))
            columna += 1
        fig.savefig(datainfo.dpath + '/' + datainfo.name + '/Results/' + datainfo.name + '-' + sensor + '-clustersNvsNP.pdf', orientation='landscape', format='pdf')
        # plt.show()
        plt.close()


        datainfo.close_experiment_data(f)

