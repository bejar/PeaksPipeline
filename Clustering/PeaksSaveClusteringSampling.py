"""
.. module:: PeaksClusteringSampling

PeaksClusteringSampling
*************

:Description: PeaksClusteringSampling

    Cluster all the peaks of the experiment using a sample of the peaks for each file

:Authors: bejar
    

:Version: 

:Created on: 11/03/2016 8:41 

"""


import numpy as np
from sklearn.cluster import KMeans

from Config.experiments import experiments
from collections import Counter
from operator import itemgetter
from util.plots import show_vsignals
import argparse

__author__ = 'bejar'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()
    lexperiments = args.exp
    nchoice = 2

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511''e110906o'
        lexperiments = ['e110906o']

    for expname in lexperiments:
        datainfo = experiments[expname]

        f = datainfo.open_experiment_data(mode='r+')

        for sensor, nclusters in zip(datainfo.sensors, datainfo.clusters):
            print(sensor)
            ldata = []
            for dfile in datainfo.datafiles:
                data = datainfo.get_peaks_resample_PCA(f, dfile, sensor)
                if data is not None:
                    idata = np.random.choice(range(data.shape[0]), data.shape[0]/nchoice, replace=False)
                    ldata.append(data[idata, :])

            data = np.vstack(ldata)
            print(data.shape)
            km = KMeans(n_clusters=nclusters, n_jobs=-1)
            km.fit(data)
            lsignals = []
            cnt = Counter(list(km.labels_))

            lmax = []
            for i in range(km.n_clusters):
                lmax.append((i, np.max(km.cluster_centers_[i])))
            lmax = sorted(lmax, key=itemgetter(1))

            centers = np.zeros(km.cluster_centers_.shape)
            for nc in range(nclusters):
                centers[nc] = km.cluster_centers_[lmax[nc][0]]

            # show_vsignals(centers)
            datainfo.save_peaks_global_clustering_centroids(f, sensor, centers)

        datainfo.close_experiment_data(f)
