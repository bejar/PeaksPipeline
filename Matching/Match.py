"""
.. module:: Match

Match
*************

:Description: Match

    

:Authors: bejar
    

:Version: 

:Created on: 27/01/2016 16:22 

"""
from munkres import Munkres
import networkx as nx
import h5py
from Config.experiments import experiments
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from util.misc import choose_color
from pyx import *
import argparse

__author__ = 'bejar'


def compute_signals_matching(datainfo, lsensors, rescale=True, globalc=False):
    """
    Computes the matching among the cluster centroids of all the signals
    :return:
    """
    def all_different(gr):
        """
        Tests if all the classes in the graph come from a different signal
        :param gr:
        :return:
        """
        sig = set()
        alldiff = True
        for node in gr:
            if not node[0:4] in sig:
                sig.add(node[0:4])
            else:
                alldiff = False
        return alldiff

    def listify(mat):
        """
        Transforms an numpy matrix to list of lists
        :param mat:
        :return:
        """
        lmat = []
        for i in range(mat.shape[0]):
            lrow = []
            for j in range(mat.shape[1]):
                lrow.append(mat[i,j])
            lmat.append(lrow)
        return lmat

    nsignals = len(lsensors)

    f = datainfo.open_experiment_data(mode='r')
    dfile = datainfo.datafiles[0]
    ename = datainfo.expnames[0]

    lcenters = []

    for sensor in datainfo.sensors:
        if globalc:
            lcenters.append(datainfo.get_peaks_global_clustering_centroids(f, sensor, datainfo.clusters[0]))
        else:
            lcenters.append(datainfo.get_peaks_clustering_centroids(f, dfile, sensor, datainfo.clusters[0]))

    lscales = []
    for cent in lcenters:
        cmx = np.max(cent)
        cmn = np.min(cent)
        lscales.append(cmx - cmn)

    lassoc = []
    for sensor1 in lsensors:
        for sensor2 in lsensors:
            s1 = datainfo.sensors.index(sensor1)
            s2 = datainfo.sensors.index(sensor2)

            if s1 < s2:
                if rescale:
                    proportion = lscales[s1] / lscales[s2]
                    centers1 = lcenters[s1]
                    centers2 = lcenters[s2] * proportion
                else:
                    centers1 = lcenters[s1]
                    centers2 = lcenters[s2]

                dist = euclidean_distances(centers1, centers2)
                dist = listify(dist)

                m = Munkres()
                indexes = m.compute(dist)

                lhun = [(row,column,dist[row][column]) for row, column in indexes]
                lassoc.append((sensor1, sensor2, lhun))

    ledges = []
    signalGraph = nx.Graph()
    for sensor1, sensor2, assoc in lassoc:
        for s1, s2, dist in assoc:
            signalGraph.add_weighted_edges_from([(sensor1+str(s1), sensor2+str(s2), dist)])
            ledges.append([sensor1+str(s1), sensor2+str(s2), dist])
    ledges = sorted(ledges, key=lambda edge: edge[2], reverse=True)

    lgraphs = []

    # Remove connected components of size less or equal than the number of signals
    # and have classes from different signals
    if nx.number_connected_components(signalGraph) > 1:
        ccomponents = [c for c in nx.connected_components(signalGraph)]
        for cc in ccomponents:
            if len(cc) <= nsignals and all_different(cc):
                lgraphs.append(cc)
                for node in cc:
                    signalGraph.remove_node(node)

    end = False
    # Remove larger edge and compute connected components until graph is empty
    while not end:
        if signalGraph.has_edge(ledges[0][0], ledges[0][1]):
            signalGraph.remove_edge(ledges[0][0], ledges[0][1])
            if nx.number_connected_components(signalGraph) > 1:
                ccomponents = [c for c in nx.connected_components(signalGraph)]
                for cc in ccomponents:
                    if len(cc) <= nsignals:
                        lgraphs.append(cc)
                        for node in cc:
                            signalGraph.remove_node(node)
            end = signalGraph.number_of_nodes() == 0

        ledges = ledges[1:]  # pop the edge

    lgraphsrt = []
    for i, gr in enumerate(lgraphs):
        lmax = []
        for pk in gr:
            sensor1 = pk[0:4]
            ncl = int(pk[4:])
            s1 = datainfo.sensors.index(sensor1)
            lmax.append(np.max(lcenters[s1][ncl]))

        lgraphsrt.append([gr, np.max(lmax)])

    datainfo.close_experiment_data(f)

    return [gr for gr, _ in sorted(lgraphsrt, key=lambda edge: edge[1])]


def compute_matching_mapping(ncl, sensor, matching):
    """
    Computes the mapping among the sensor clusters and the sensors matching
    :param matching:
    :param ncl:
    :param sensor:
    :return:
    """
    mapping = np.zeros(ncl)
    for i in range(ncl):
        for j, m in enumerate(matching):
            if sensor+str(i) in m:
                mapping[i] = j
    return mapping


def save_matching(matching, lsensors, rescale=True):
    """

    :param smatching:
    :return:
    """

    collist = choose_color(len(matching))

    lpages = []

    c = canvas.canvas()
    c.text(0, (len(matching)+1)*5, "%s" % (datainfo.name), [text.size(5)])

    for i, s in enumerate(lsensors):
        c.text((i*5)+1, len(matching)*5, s, [text.size(5)])
        c.text((i*5)+1, -2, s, [text.size(5)])

    for i, match in enumerate(matching):
        p = path.rect(-5.5, i*5,  5, 5)
        c.stroke(p, [deco.filled([collist[i]])])
        peaks = [False] * len(lsensors)
        nclust = [''] * len(lsensors)
        for m in match:
            if m[0:4] in lsensors:
                peaks[lsensors.index(m[0:4])] = True
                nclust[lsensors.index(m[0:4])] = str(int(m[4:])+1)

        for j in range(len(lsensors)):
            if peaks[j]:
                isens = datainfo.sensors.index(lsensors[j])
                fname = datainfo.dpath + '/' + datainfo.name + "/Results/icons/" + datainfo.name + lsensors[j] \
                        + '.nc' + str(datainfo.clusters[isens]) + '.cl' + nclust[j] + '.jpg'
                bm = bitmap.jpegimage(fname)
                bm.info = {}
                bm.info['dpi'] = (100, 100)
                bm2 = bitmap.bitmap(j*5, i*5, bm, compressmode=None)
                c.insert(bm2)

    lpages.append(document.page(c))

    d = document.document(lpages)
    res = ''
    if rescale:
        res = 'rescale'

    d.writePDFfile(datainfo.dpath + '/' + datainfo.name + "/Results/peaksmatching-%s-%s" % (datainfo.name, res))


if __name__ == '__main__':

    # Matching parameters
    isig = 2
    fsig = 10

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--rescale', help="Reescala proporcionalmente el matching", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")
    parser.add_argument('--globalclust', help="Use a global computed clustering", action='store_true', default=False)

    args = parser.parse_args()

    lexperiments = args.exp

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511''e160204''e110906o'
        lexperiments = ['e150514']
        args.rescale = True
        args.globalclust = True

    for expname in lexperiments:

        datainfo = experiments[expname]

        lsensors = datainfo.sensors[isig:fsig]
        lclusters = datainfo.clusters[isig:fsig]
        smatching = compute_signals_matching(datainfo, lsensors, rescale=args.rescale, globalc=args.globalclust)
        save_matching(smatching, lsensors, rescale=args.rescale)