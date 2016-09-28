"""
.. module:: Outliers

Outliers
*************

:Description: Outliers

    

:Authors: bejar
    

:Version: 

:Created on: 19/09/2016 13:26 

"""

__author__ = 'bejar'

from sklearn.neighbors import NearestNeighbors
import numpy as np

def outliers_knn(data, nn, nstd=6):
    """
    Identifies the outliers in the peaks

    :param dfile:
    :param sensor:
    :return:
    """

    # Detect outliers based on the distribution of the distances of the signals to the knn
    # Any signal that is farther from its neighbors that a number of standard deviations of the mean knn-distance is out
    neigh = NearestNeighbors(n_neighbors=nn)
    neigh.fit(data)

    vdist = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        vdist[i] = np.sum(neigh.kneighbors(data[i].reshape(1, -1), return_distance=True)[0][0][1:]) / (nn - 1)
    dmean = np.mean(vdist)
    dstd = np.std(vdist)
    return vdist < dmean + (nstd * dstd)

def outliers_wavy(data, wavy=5):
    """
    selects too wavy signals

    :param datainfo:
    :param dfile:
    :param sensor:
    :param wavy:
    :return:
    """
    def is_wavy_signal(signal, thresh):
        """
        Detects if a signal has many cuts over the positive middle part of the signal

        :param signal:
        :return:
        """

        middle = np.max(signal) * 0.3
        tmp = signal.copy()

        tmp[signal < middle] = 0

        count = 0
        for i in range(1, signal.shape[0]):
            if tmp[i - 1] == 0 and tmp[i] != 0:
                count += 1
        return count > thresh

    vsel = np.ones(data.shape[0],dtype=bool)
    for i in range(data.shape[0]):
        vsel[i] = not is_wavy_signal(data[i], wavy)

    return vsel
