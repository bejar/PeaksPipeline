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
import seaborn as sn
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

warnings.filterwarnings("ignore")


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

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511''e150514''e150514alt', 'e150514'
        args.hellinger = False
        lexperiments = ['e110906o']
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

            lsignals = []

            data = datainfo.get_peaks_resample(f, datainfo.datafiles[0], sensor)
            labels = datainfo.compute_peaks_labels(f, datainfo.datafiles[0], sensor, nclusters, globalc=args.globalclust)

            peaks_std = np.zeros((nclusters, data.shape[1]))
            peaks_min = np.zeros((nclusters, data.shape[1]))
            peaks_max = np.zeros((nclusters, data.shape[1]))

            for nc in range(nclusters):
                peaks_min[nc] = np.min(data[labels == nc], axis=0)
                peaks_max[nc] = np.max(data[labels == nc], axis=0)
                peaks_std[nc] = np.std(data[labels == nc], axis=0)

            # mhisto = np.zeros((len(datainfo.datafiles)//batches, nclusters))
            # cbatch = [c for c in enumerate(datainfo.datafiles)]
            # lbatch = batchify(cbatch, batches)
            #
            # for nf, btch in enumerate(lbatch):
            #     npeaks = 0
            #     histo = np.zeros(nclusters)
            #     for _, dfile in btch:
            #
            #         labels = datainfo.compute_peaks_labels(f, dfile, sensor, nclusters, globalc=args.globalclust)
            #         npeaks += len(labels)
            #         for i in labels:
            #             mhisto[nf, i] += 1.0
            #     mhisto[nf] /= npeaks

            matplotlib.rcParams.update({'font.size': 25})
            fig = plt.figure()
            fig.set_figwidth(15)
            fig.set_figheight(9)
            width = 1

            ncols = 3
            nrows = 5


            minaxis = np.min(centroids+peaks_std)
            maxaxis = np.max(centroids-peaks_std)

            minaxis = np.min(peaks_min)
            maxaxis = np.max(peaks_max)

            sepy = round((maxaxis - minaxis)/4, 2)

            for nc in range(nclusters):
                ax2 = fig.add_subplot(ncols, nrows, nc+1)


                signal = centroids[nc]
                lenplot = datainfo.peaks_resampling['wtsel']
                t = np.arange(0.0, len(signal), 1)/len(signal) * 100
                ax2.axis([0, lenplot, minaxis, maxaxis])
                ax2.plot(t,signal, linewidth=2)
                ax2.xaxis.set_major_locator(ticker.MultipleLocator(50))
                ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

                ax2.plot(t,signal+peaks_std[nc], color='g',linewidth=0.5, linestyle=':')
                ax2.plot(t,signal-peaks_std[nc], color='g',linewidth=0.5, linestyle=':')

                ax2.annotate(str(nc+1), xy=(0, 0), xycoords='data',
                xytext=(0.95, 0.98), textcoords='axes fraction',
                horizontalalignment='right', verticalalignment='top', fontsize=16
                )

                if nc == 10:
                    plt.ylabel('millivolts', fontsize=18)
                    plt.xlabel('time(ms)', fontsize=18)
                plt.axhline(linewidth=1, color='r', y=0)


            fig.savefig(datainfo.dpath + '/' + datainfo.name + '/Results/' + datainfo.name + '-' + sensor + '-' +
                        str(nclusters) + '-peaks-fig.pdf', orientation='landscape', format='pdf')
            #plt.show()
