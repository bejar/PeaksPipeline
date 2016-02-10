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
import networkx as nx

__author__ = 'bejar'


def compute_experiments_graph(experiments, mdist, sensor, knn=1):
    """
    Computes a graph with the k-nn links according to the probability matrices distances
    :return:
    """

    experimentsGraph=nx.Graph()

    for i, nex in enumerate(experiments):
        ldist = []

        for j, dist in enumerate(mdist[i]):
            ldist.append((dist,j))
        ldist = sorted(ldist, key=lambda edge: edge[0], reverse=True)
        for j in range(knn):
            experimentsGraph.add_weighted_edges_from([(experiments[i][8:], experiments[ldist[j][1]][8:], ldist[j][0])])

    pos=nx.graphviz_layout(experimentsGraph)
    nx.draw_networkx_nodes(experimentsGraph, pos)
    nx.draw_networkx_edges(experimentsGraph, pos)
    nx.draw_networkx_labels(experimentsGraph, pos)
    #nx.draw_graphviz(experimentsGraph)
    plt.show()


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
        print pm
        return pm

# --------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()
    lexperiments = args.exp

    if not args.exp:
        # 'e150514''e120503''e110616''e150707''e151126''e120511''e110906e'
        lexperiments = ['e150514']


    norm = 'Row' # 'Row', 'All'

    peakdata = {}
    for expname in lexperiments:

        datainfo = experiments[expname]
        f = h5py.File(datainfo.dpath + '/' + datainfo.name + '/' + datainfo.name + '.hdf5', 'r')

        for ncl, sensor in zip(datainfo.clusters[6:10], datainfo.sensors[6:10]):
            print(sensor)
            lmatrix = []
            for dfile, ename in zip(datainfo.datafiles, datainfo.expnames):
                clpeaks = compute_data_labels(datainfo.datafiles[0], dfile, sensor)
                d = f[dfile + '/' + sensor + '/' + 'Time']
                timepeaks =  d[()]

                mtrx = generate_prob_matrix( timepeaks, clpeaks, ncl, gap=2000, laplace=0, norm= norm)
                lmatrix.append(mtrx)

                # fig = plt.figure()
                # plt.title(ename)
                # fig.set_figwidth(50)
                # fig.set_figheight(60)
                # sns.heatmap(mtrx, cmap='jet', vmin=0, vmax=0.25)
                # plt.show()

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
            mdist = np.zeros((len(lmatrix),  len(lmatrix)))
            for i in range(len(lmatrix)):
                for j in range(len(lmatrix)):
                    mdist[i,j] = hamming_frobenius_distance(lmatrix[i], lmatrix[j])

            compute_experiments_graph(datainfo.expnames, mdist, sensor, knn=1)