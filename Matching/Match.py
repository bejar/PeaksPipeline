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

__author__ = 'bejar'


def compute_signals_matching(expname, lsensors):
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

    nsignals = len(lsensors)


    datainfo = experiments[expname]

    f = h5py.File(datainfo.dpath + datainfo.name + '/' + datainfo.name + '.hdf5', 'r')

    for dfile, ename in zip([datainfo.datafiles[0]], [datainfo.expnames[0]]):

        lcenters = []

        for sensor in datainfo.sensors:

            d = f[dfile + '/' + sensor + '/Clustering/' + 'Centers']
            lcenters.append(d[()])

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
                    rescale = lscales[s1]/lscales[s2]
                    centers1 = lcenters[s1]
                    centers2 = lcenters[s2] * rescale

                    dist = euclidean_distances(centers1,centers2)

                    m = Munkres()
                    indexes = m.compute(dist.copy())


                    lhun=[(row,column,dist[row][column]) for row, column in indexes]
                    lassoc.append((sensor1, sensor2, lhun))
        ledges = []
        signalGraph=nx.Graph()
        for sensor1, sensor2, assoc in lassoc:
            for s1, s2, dist in assoc:
                signalGraph.add_weighted_edges_from([(sensor1+str(s1),sensor2+str(s2),dist)])
                ledges.append([sensor1+str(s1),sensor2+str(s2),dist])
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

        return [gr for gr, _ in sorted(lgraphsrt, key=lambda edge: edge[1])]


def compute_matching_mapping(ncl, sensor, matching):
    """
    Computes the mapping among the sensor clusters and the sensors matching
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