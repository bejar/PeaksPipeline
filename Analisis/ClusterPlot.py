"""
.. module:: ClusterPlot

ClusterPlot
*************

:Description: ClusterPlot

 Histogramas de los clusters para las figuras de los articulos
    

:Authors: bejar
    

:Version: 

:Created on: 12/05/2016 14:27 

"""

__author__ = 'bejar'


from collections import Counter
from operator import itemgetter

import matplotlib.pyplot as plt
from pylab import *
from sklearn.cluster import KMeans
from kemlglearn.cluster import KernelKMeans
from Config.experiments import experiments
from util.plots import plotSignals
import warnings
from util.distances import hellinger_distance
from util.misc import compute_centroids
import argparse
import matplotlib.ticker as ticker
from util.itertools import batchify
import numpy as np
import seaborn as sn

warnings.filterwarnings("ignore")

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--join', help="Joins the files in groups", type=int, nargs='+', default=1)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")
    parser.add_argument('--globalclust', help="Use a global computed clustering", action='store_true', default=False)
    args = parser.parse_args()

    lexperiments = args.exp
    batches = args.join
    plt.style.use('seaborn-darkgrid')
    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511''e150514''e150514alt', 'e150514''e130221c''e160802'
        args.hellinger = False
        lexperiments = ['e150514']
        batches = 1

    for expname in lexperiments:
        datainfo = experiments[expname]
        colors = [datainfo.colors[i] for i in range(0, len(datainfo.colors), batches)]

        f = datainfo.open_experiment_data(mode='r')

        for sensor, nclusters in zip(datainfo.sensors, datainfo.clusters):
            print(sensor, nclusters)

            if args.globalclust:
                centroids = datainfo.get_peaks_global_clustering_centroids(f, sensor, nclusters)
            else:
                centroids = datainfo.get_peaks_clustering_centroids(f, datainfo.datafiles[0], sensor, nclusters)
                variance = np.zeros(centroids.shape)
                labels = datainfo.compute_peaks_labels(f, datainfo.datafiles[0], sensor, nclusters, globalc=args.globalclust)
                data = datainfo.get_peaks_resample_PCA(f, datainfo.datafiles[0], sensor)
                for i in range(nclusters):
                    variance[i] = np.std(data[labels==i], axis=0)


            lsignals = []

            mhisto = np.zeros((len(datainfo.datafiles)//batches, nclusters))
            cbatch = [c for c in enumerate(datainfo.datafiles)]
            lbatch = batchify(cbatch, batches)
            for nf, btch in enumerate(lbatch):
                npeaks = 0
                histo = np.zeros(nclusters)
                for _, dfile in btch:

                    labels = datainfo.compute_peaks_labels(f, dfile, sensor, nclusters, globalc=args.globalclust)
                    npeaks += len(labels)
                    for i in labels:
                        mhisto[nf, i] += 1.0
                mhisto[nf] /= npeaks



            matplotlib.rcParams.update({'font.size': 15})
            fig = plt.figure()
            fig.set_figwidth(24)
            fig.set_figheight(18)
            width = 1
            ncols = nclusters / 2
            if nclusters % 2 == 1:
                ncols += 1
            for i in range(nclusters):

                ax = fig.add_subplot(ncols, 4, (i*2)+2)

                ax.axis([0, mhisto.shape[0], 0, 0.501])
                rects = ax.bar(range(mhisto.shape[0]), mhisto[: , i], width, color=colors, linewidth=1)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))

            minaxis = np.min(centroids - variance)
            maxaxis = np.max(centroids + variance)
            sepy = round((maxaxis - minaxis)/4, 2)

            for nc in range(nclusters):
                ax2 = fig.add_subplot(ncols, 4, (nc*2)+1)
                signal = centroids[nc]
                signalv = variance[nc]
                plt.text(10,maxaxis*.8, LETTERS[nc], fontsize=20)
                lenplot = datainfo.peaks_resampling['wtsel']
                t = np.arange(0.0, len(signal), 1)/len(signal) * 100
                ax2.axis([0, lenplot, minaxis, maxaxis])
                ax2.plot(t,signal)
                #ax2.plot(t,signal + signalv, c='g')
                #ax2.plot(t,signal -  signalv, c='g')
                ax2.xaxis.set_major_locator(ticker.MultipleLocator(50))
                ax2.yaxis.set_major_locator(ticker.MultipleLocator(sepy))
                plt.axhline(linewidth=1, color='r', y=0)

            fig.savefig(datainfo.dpath + '/' + datainfo.name + '/Results/' + datainfo.name + '-' + sensor + '-' +
                        str(nclusters) + '-histo-fig.pdf', orientation='landscape', format='pdf')
            #plt.show()
