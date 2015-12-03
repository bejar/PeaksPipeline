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
from util.plots import plotSignals
import warnings
warnings.filterwarnings("ignore")

__author__ = 'bejar'
# 'e110616''e120503''e150514'
lexperiments = ['e150707']


for expname in lexperiments:
    datainfo = experiments[expname]
    colors = datainfo.colors

    f = h5py.File(datainfo.dpath + datainfo.name + '/' + datainfo.name + '.hdf5', 'r')

    for s, nclusters in zip(datainfo.sensors, datainfo.clusters):
        print(s)
        ldata = []
        for dfiles in datainfo.datafiles:
            d = f[dfiles + '/' + s + '/' + 'PeaksResamplePCA']
            dataf = d[()]
            ldata.append(dataf)

        data = ldata[0] #np.concatenate(ldata)

        km = KMeans(n_clusters=nclusters, n_jobs=-1)
        km.fit_transform(data)
        lsignals = []
        cnt = Counter(list(km.labels_))

        lmax = []
        for i in range(km.n_clusters):
            lmax.append((i,np.max(km.cluster_centers_[i])))
        lmax = sorted(lmax, key=itemgetter(1))

        print('LMAX ', lmax)
        print('SHAPE ', data.shape)

        lhisto = []
        for dataf, ndata in zip(ldata, datainfo.datafiles):
            histo = np.zeros(nclusters)
            for i in range(dataf.shape[0]):
                histo[km.predict(dataf[i])] += 1.0
            histo /= dataf.shape[0]
            print(datainfo.name, ndata)
            print('HISTO ', histo)
            histosorted = np.zeros(nclusters)
            for i in range(histosorted.shape[0]):
                histosorted[i] = histo[lmax[i][0]]
            lhisto.append(histosorted)

        # for h in lhisto[1:]:
        #     rms = np.dot(lhisto[0] - h,  lhisto[0] - h)
        #     rms /= h.shape[0]
        #     print np.sqrt(rms), hellinger_distance(h, lhisto[0])

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
        fig.suptitle(datainfo.name + '-' + s, fontsize=48)

        minaxis = np.min(km.cluster_centers_)
        maxaxis = np.max(km.cluster_centers_)


        for nc in range(nclusters):
            ax2 = fig.add_subplot(2, nclusters, nc+nclusters+1)
            signal = km.cluster_centers_[lmax[nc][0]]
            plt.title(' ( '+str(cnt[lmax[nc][0]])+' )')
            t = arange(0.0, len(signal), 1)
            ax2.axis([0, len(signal), minaxis, maxaxis])
            ax2.plot(t,signal)
            plt.axhline(linewidth=1, color='r', y=0)
        fig.savefig(datainfo.dpath + '/' + datainfo.name + '/Results/' + datainfo.name + '-' + s + '-histo-sort.pdf', orientation='landscape', format='pdf')
    #    plt.show()

        print('*******************')
        for nc in range(nclusters):
            lsignals.append((km.cluster_centers_[lmax[nc][0]], str(nc)+' ( '+str(cnt[lmax[nc][0]])+' )'))

        if nclusters % 2 == 0:
            part = nclusters /2
        else:
            part = (nclusters /2) + 1
        plotSignals(lsignals,part,2,np.max(km.cluster_centers_),np.min(km.cluster_centers_), datainfo.name + '-' + s,
                    datainfo.name + '-' + s, datainfo.dpath+ '/' + datainfo.name + '/Results/')
    f.close()