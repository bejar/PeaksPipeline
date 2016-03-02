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


__author__ = 'bejar'
# 'e150514''e120503'
lexperiments = ['e110906e']


def show_vsignals(signal, centroid, title=''):
    """
    Plots a list of signals
    :param signal:
    :return:
    """
    matplotlib.rcParams.update({'font.size': 26})
    fig = plt.figure()
    fig.set_figwidth(30)
    fig.set_figheight(40)
    fig.suptitle(title, fontsize=48)
    minaxis = np.min(signal)
    maxaxis = np.max(signal)
    num = signal.shape[1]
    sp1 = fig.add_subplot(111)
    sp1.axis([0, num, minaxis, maxaxis])
    t = arange(0.0, num, 1)
    for i in range(signal.shape[0]):
        sp1.plot(t, signal[i,:])
    sp1.plot(t, centroid, color='r', linewidth=8.0)

    plt.show()


for expname in lexperiments:
    datainfo = experiments[expname]
    colors = datainfo.colors

    f = h5py.File(datainfo.dpath + datainfo.name + '/' + datainfo.name + '.hdf5', 'r')

    for s, nclusters in zip([datainfo.sensors[5]], [datainfo.clusters[5]]):
        print(s)
        ldata = []
        for dfiles in [datainfo.datafiles[0]]:
            d = f[dfiles + '/' + s + '/' + 'PeaksResample']
            dataf = d[()]
            ldata.append(dataf)

        data = ldata[0] #np.concatenate(ldata)

        km = KMeans(n_clusters=nclusters)
        km.fit_transform(data)

        cnt = Counter(list(km.labels_))
        for i in np.unique(km.labels_):
            print len(dataf[km.labels_ == i, :])
            show_vsignals(dataf[km.labels_ == i, :], km.cluster_centers_[i])


    f.close()