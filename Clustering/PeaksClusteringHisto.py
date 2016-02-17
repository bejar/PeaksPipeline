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
warnings.filterwarnings("ignore")

__author__ = 'bejar'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()

    lexperiments = args.exp

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511''e150514'
        lexperiments = ['e151126']


    for expname in lexperiments:
        datainfo = experiments[expname]
        colors = datainfo.colors

        f = datainfo.open_experiment_data(mode='r')

        for sensor, nclusters in zip(datainfo.sensors, datainfo.clusters):
            print(sensor)
            ldata = []
            for dfile in datainfo.datafiles:
                ldata.append(datainfo.get_peaks_resample_PCA(f, dfile, sensor))

            data = ldata[0] #np.concatenate(ldata)

            #km = KernelKMeans(n_clusters=nclusters, kernel='rbf', degree=2, gamma=0.05)
            km = KMeans(n_clusters=nclusters, n_jobs=-1)
            km.fit_predict(data)
            centroids = km.cluster_centers_
            #centroids = compute_centroids(data, km.labels_)

            lsignals = []
            cnt = Counter(list(km.labels_))

            lmax = []
            for i in range(km.n_clusters):
                lmax.append((i, np.max(centroids[i])))
            lmax = sorted(lmax, key=itemgetter(1))

            print('LMAX ', lmax)
            print('SHAPE ', data.shape)

            lhisto = []
            for dataf, ndata in zip(ldata, datainfo.datafiles):
                if dataf is not None:
                    histo = np.zeros(nclusters)
                    for i in range(dataf.shape[0]):
                        histo[km.predict(dataf[i])] += 1.0
                    histo /= dataf.shape[0]
                    print(datainfo.name, ndata)
                    print('HISTO ', histo)
                    histosorted = np.zeros(nclusters)
                    for i in range(histosorted.shape[0]):
                        histosorted[i] = histo[lmax[i][0]]
                else:
                    histosorted = np.zeros(nclusters)
                lhisto.append(histosorted)


            for h in lhisto[1:]:
                rms = np.dot(lhisto[0] - h,  lhisto[0] - h)
                rms /= h.shape[0]
                print(np.sqrt(rms), hellinger_distance(h, lhisto[0]))

            matplotlib.rcParams.update({'font.size': 30})
            fig = plt.figure()
            ax = fig.add_subplot(2, 1, 1)
            fig.set_figwidth(60)
            fig.set_figheight(40)
            ind = np.arange(nclusters)  # the x locations for the groups
            width = 1.0/(len(lhisto)+1)   # the width of the bars
            ax.set_xticks(ind+width)
            ax.set_xticklabels(ind)
            for i, h in enumerate(lhisto):
                rects = ax.bar(ind+(i*width), h, width, color=colors[i])
            fig.suptitle(datainfo.name + '-' + sensor, fontsize=48)

            minaxis = np.min(centroids)
            maxaxis = np.max(centroids)


            for nc in range(nclusters):
                ax2 = fig.add_subplot(2, nclusters, nc+nclusters+1)
                signal = centroids[lmax[nc][0]]
                plt.title(' ( '+str(cnt[lmax[nc][0]])+' )')
                t = arange(0.0, len(signal), 1)
                ax2.axis([0, len(signal), minaxis, maxaxis])
                ax2.plot(t,signal)
                plt.axhline(linewidth=1, color='r', y=0)
            fig.savefig(datainfo.dpath + '/' + datainfo.name + '/Results/' + datainfo.name + '-' + sensor + '-histo-sort.pdf', orientation='landscape', format='pdf')
        #    plt.show()

            print('*******************')
            for nc in range(nclusters):
                lsignals.append((centroids[lmax[nc][0]], str(nc)+' ( '+str(cnt[lmax[nc][0]])+' )'))

            if nclusters % 2 == 0:
                part = nclusters /2
            else:
                part = (nclusters /2) + 1
            plotSignals(lsignals, part, 2, maxaxis, minaxis, datainfo.name + '-' + sensor,
                        datainfo.name + '-' + sensor, datainfo.dpath + '/' + datainfo.name + '/Results/')
        datainfo.close_experiment_data(f)