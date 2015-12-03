"""
.. module:: PeaksClustering

PeaksClustering
*************

:Description: PeaksClustering

    Generates and saves a clustering of the peaks for all the files of the experiment

:Authors: bejar
    

:Version: 

:Created on: 26/03/2015 8:10 

"""

__author__ = 'bejar'

import h5py
import numpy as np
from sklearn.cluster import KMeans

from Config.experiments import experiments
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from operator import itemgetter


# 'e110616''e120503''e150514'
lexperiments = ['e150707']


for expname in lexperiments:
    datainfo = experiments[expname]

    f = h5py.File(datainfo.dpath + datainfo.name + '/' + datainfo.name + '.hdf5', 'r+')

    for sensor, nclusters in zip(datainfo.sensors, datainfo.clusters):
        print(sensor)
        ldata = []
        for dfile in datainfo.datafiles:
            print(dfile)
            d = f[dfile + '/' + sensor + '/' + 'PeaksResamplePCA']
            data = d[()]

            km = KMeans(n_clusters=nclusters, n_jobs=-1)
            km.fit_transform(data)
            lsignals = []
            cnt = Counter(list(km.labels_))

            lmax = []
            for i in range(km.n_clusters):
                lmax.append((i,np.max(km.cluster_centers_[i])))
            lmax = sorted(lmax, key=itemgetter(1))

            centers = np.zeros(km.cluster_centers_.shape)
            for nc in range(nclusters):
                centers[nc] = km.cluster_centers_[lmax[nc][0]]
            if dfile + '/' + sensor + '/Clustering/Centers' in f:
                del f[dfile + '/' + sensor + '/Clustering/Centers']
            d = f.require_dataset(dfile + '/' + sensor + '/Clustering/' + 'Centers', centers.shape, dtype='f',
                                  data=centers, compression='gzip')
            d[()] = centers
            lmax = []
    f.close()

