"""
.. module:: PeaksSequencesClustering

PeaksSequencesClustering
******

:Description: PeaksSequencesClustering

    Calcula un clustering jerarquico con las matrices de transicion correspondientes a los eventos de las secuencias
    de los diferentes sensores

:Authors:
    bejar

:Version: 

:Date:  21/12/2015
"""

import operator
import h5py
import numpy as np

from Config.experiments import experiments
import scipy.io
from pylab import *

from util.distances import square_frobenius_distance, hellinger_distance, bhattacharyya_distance, hamming_frobenius_distance
from sklearn.metrics import pairwise_distances_argmin_min

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import seaborn as sns
import argparse

__author__ = 'bejar'

def compute_data_labels(dfilec, dfile, sensor):
    """
    Computes the labels of the data using the centroids of the cluster in the file
    :param dfile:
    :param sensor:
    :return:
    """
    f = h5py.File(datainfo.dpath + '/' + datainfo.name + '/' + datainfo.name  + '.hdf5', 'r')

    d = f[dfilec + '/' + sensor + '/Clustering/' + 'Centers']
    centers = d[()]
    d = f[dfile + '/' + sensor + '/' + 'PeaksResamplePCA']
    data = d[()]
    labels, _ = pairwise_distances_argmin_min(data, centers)
    f.close()
    return labels



def generate_prob_matrix(timepeaks, clpeaks, nsym, gap, laplace=0.0, norm='All'):
    """
    Computes the transition probability matrix from the times of the peaks considering that
    gap is the minimum time between consecutive peaks that indicates a pause (time in the sampling frequency)

    :param dfile:
    :param timepeaks:
    :param clpeaks:
    :param sensor:
    :param norm:
    :return:
    """
    pm = np.zeros((nsym, nsym)) + laplace
    for i in range(timepeaks.shape[0]-1):
        if timepeaks[i + 1] - timepeaks[i] < gap:
            pm[clpeaks[i], clpeaks[i + 1]] += 1.0
    if norm == 'All':
        return pm / np.sum(pm)
    else:
        for i in range(nsym):
            pm[i] /= np.sum(pm[i])
        return pm

# --------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # 'e150514''e120503''e110616''e150707''e151126''e120511'
    lexperiments = ['e120511']

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()
    if args.exp:
        lexperiments = args.exp


    norm = 'Row' # 'Row'

    peakdata = {}
    for expname in lexperiments:

        datainfo = experiments[expname]
        f = h5py.File(datainfo.dpath + '/' + datainfo.name + '/' + datainfo.name + '.hdf5', 'r')

        for ncl, sensor in zip(datainfo.clusters[3:], datainfo.sensors[3:]):
            print(sensor)
            lmatrix = []
            for dfile, ename in zip(datainfo.datafiles, datainfo.expnames):
                clpeaks = compute_data_labels(datainfo.datafiles[0], dfile, sensor)
                d = f[dfile + '/' + sensor + '/' + 'Time']
                timepeaks =  d[()]

                mtrx = generate_prob_matrix( timepeaks, clpeaks, ncl, gap=2000, laplace=1, norm= norm)
                lmatrix.append(mtrx)

                fig = plt.figure()
                fig.set_figwidth(50)
                fig.set_figheight(60)
                sns.heatmap(mtrx, cmap='jet')
                plt.show()

            # distance matrix among probability matrices
            mdist = np.zeros(len(lmatrix) * (len(lmatrix)-1) / 2)
            pos = 0
            for i in range(len(lmatrix)):
                for j in range(0, i):
                    mdist[pos] = hamming_frobenius_distance(lmatrix[i], lmatrix[j])
                    pos += 1

            clust = linkage(mdist, method='single')

            plt.figure(figsize=(15,15))
            dendrogram(clust, distance_sort=True, orientation='left', labels=datainfo.expnames)
            plt.show()
