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
from util.plots import plotSignals, show_vsignals

__author__ = 'bejar'
# 'e150514''e120503'
lexperiments = ['e150514']


for expname in lexperiments:
    datainfo = experiments[expname]
    colors = datainfo.colors

    f = h5py.File(datainfo.dpath + datainfo.name + '/' + datainfo.name + '.hdf5', 'r')

    for s, nclusters in zip([datainfo.sensors[0]], [datainfo.clusters[0]]):
        print(s)
        ldata = []
        for dfiles in [datainfo.datafiles[0]]:
            d = f[dfiles + '/' + s + '/' + 'PeaksFilter']
            dataf = d[()]
            ldata.append(dataf)

        data = ldata[0] #np.concatenate(ldata)

        km = KMeans(n_clusters=nclusters)
        km.fit_transform(data)

        cnt = Counter(list(km.labels_))
        for i in np.unique(km.labels_):
            print len(dataf[km.labels_ == i, :])
            show_vsignals(dataf[km.labels_ == i, :])


    f.close()