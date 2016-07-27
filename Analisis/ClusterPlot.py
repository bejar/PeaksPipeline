"""
.. module:: ClusterPlot

ClusterPlot
*************

:Description: ClusterPlot

    

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

warnings.filterwarnings("ignore")


__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")
    parser.add_argument('--globalclust', help="Use a global computed clustering", action='store_true', default=False)
    args = parser.parse_args()

    lexperiments = args.exp


    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511''e150514'
        args.hellinger = False
        lexperiments = ['e120511c']

    for expname in lexperiments:
        datainfo = experiments[expname]
        colors = datainfo.colors

        f = datainfo.open_experiment_data(mode='r')

        for sensor, nclusters in zip(datainfo.sensors, datainfo.clusters):
            print(sensor, nclusters)

            if args.globalclust:
                centroids = datainfo.get_peaks_global_clustering_centroids(f, sensor, nclusters)
            else:
                centroids = datainfo.get_peaks_clustering_centroids(f, datainfo.datafiles[0], sensor, nclusters)

            lsignals = []

            mhisto = np.zeros((len(datainfo.datafiles), nclusters))
            for nf, dfile in enumerate(datainfo.datafiles):

                labels = datainfo.compute_peaks_labels(f, dfile, sensor, nclusters, globalc=args.globalclust)

                histo = np.zeros(nclusters)
                for i in labels:
                    mhisto[nf, i] += 1.0
                mhisto[nf] /= len(labels)

            matplotlib.rcParams.update({'font.size': 25})
            fig = plt.figure()
            fig.set_figwidth(24)
            fig.set_figheight(12)
            width = 1
            for i in range(nclusters):

                ax = fig.add_subplot(nclusters/2, 4, (i*2)+2)

                ax.axis([0, mhisto.shape[0], 0, 0.5])
                rects = ax.bar(range(mhisto.shape[0]), mhisto[: , i], width, color=colors)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))

            minaxis = np.min(centroids)
            maxaxis = np.max(centroids)
            sepy = round((maxaxis - minaxis)/4, 2)

            for nc in range(nclusters):
                ax2 = fig.add_subplot(nclusters/2, 4, (nc*2)+1)
                signal = centroids[nc]
                #plt.title(' ( '+str(nc+1)+' )')
                t = arange(0.0, len(signal), 1)
                ax2.axis([0, len(signal), minaxis, maxaxis])
                ax2.plot(t,signal)
                ax2.xaxis.set_major_locator(ticker.MultipleLocator(50))
                ax2.yaxis.set_major_locator(ticker.MultipleLocator(sepy))
                plt.axhline(linewidth=1, color='r', y=0)

            fig.savefig(datainfo.dpath + '/' + datainfo.name + '/Results/' + datainfo.name + '-' + sensor + '-' +
                        str(nclusters) + '-histo-fig.pdf', orientation='landscape', format='pdf')
            #plt.show()
