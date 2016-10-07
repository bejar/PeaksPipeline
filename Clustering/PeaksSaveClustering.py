"""
.. module:: PeaksClustering

PeaksClustering
*************

:Description: PeaksClustering

    Generates and saves a clustering of the peaks for all the files of the experiment

:Authors: bejar
    

:Version: 

:Created on: 26/03/2015 8:10

* 4/2/2016 - Adapting to changes in Experiment class

"""

__author__ = 'bejar'

import numpy as np
from sklearn.cluster import KMeans

from Config.experiments import experiments
from collections import Counter
from operator import itemgetter
from Preproceso.Outliers import outliers_knn
import argparse

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--outliers', help="Elimina outliers de los datos", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()
    lexperiments = args.exp

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511''e160204''e160317''e110906o''e120511''e140225'
        lexperiments = ['e130221rl']
        args.outliers = True

    for expname in lexperiments:
        datainfo = experiments[expname]

        f = datainfo.open_experiment_data(mode='r+')

        for sensor, nclusters in zip(datainfo.sensors, datainfo.clusters):
            print(sensor)
            #ldata = []
            for dfile in datainfo.datafiles:
                print(dfile)
                data = datainfo.get_peaks_resample_PCA(f, dfile, sensor)
                if data is not None:
                    if data.shape[0] > nclusters:
                        km = KMeans(n_clusters=nclusters, n_jobs=-1)
                        if args.outliers:
                            data = data[outliers_knn(data, 7, nstd=5)]
                        km.fit(data)
                        lsignals = []
                        # cnt = Counter(list(km.labels_))

                        lmax = []
                        for i in range(km.n_clusters):
                            lmax.append((i, np.max(km.cluster_centers_[i])))
                        lmax = sorted(lmax, key=itemgetter(1))

                        centers = np.zeros(km.cluster_centers_.shape)
                        for nc in range(nclusters):
                            centers[nc] = km.cluster_centers_[lmax[nc][0]]

                        datainfo.save_peaks_clustering_centroids(f, dfile, sensor, centers)

                        lmax = []
        datainfo.close_experiment_data(f)

