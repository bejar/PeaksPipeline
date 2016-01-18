"""
.. module:: PeaksSyncroContingency

PeaksSyncroContingency
*************

:Description: PeaksSyncroContingency

    

:Authors: bejar
    

:Version: 

:Created on: 08/07/2015 10:04 

"""

__author__ = 'bejar'


from pyx import *
from pyx.color import cmyk, rgb, gray
from PIL import Image
from Config.experiments import experiments
from PeaksSynchro import compute_synchs, gen_data_matrix
import h5py
import numpy as np
from munkres import Munkres
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import euclidean_distances
import argparse

def peaks_contingency(peakdata, sensors, clusters):
    """
    Generates PDFs with the association frequencies of the peaks
    Each sensor with the synchronizations to the other sensors desagregated by peak class

    :return:
    """
    pk = peakdata

    dmatrix = gen_data_matrix(sensors, clusters)

    for p in pk:
        for ei in p:
            for ej in p:
                if ei[0] != ej[0]:
                    m = dmatrix[ei[0]][ej[0]]
                    m[ei[2]][ej[2]] += 1
    return dmatrix


def draw_page(c, file, sensori, sensorj, nci, ncj, matrix):
    """
    Generates the contingency table for the peaks

    :param c:
    :param file:
    :param sensori:
    :param sensorj:
    :param nci:
    :param ncj:
    :return:
    """

    for i in range(nci):
        im = Image.open(datainfo.dpath + '/' + datainfo.name +'/Results/icons/' + "%s%s.cl%d.png"%(file, sensori, i+1))
        image_bw = bitmap.image(im.size[0], im.size[1], im.mode, im.tostring())
        bitmap_bw = bitmap.bitmap(0, i+1, image_bw, height=0.8)
        c.insert(bitmap_bw)
    for i in range(ncj):
        im = Image.open(datainfo.dpath + '/' + datainfo.name +'/Results/icons/' + "%s%s.cl%d.png"%(file, sensorj, i+1))
        image_bw = bitmap.image(im.size[0], im.size[1], im.mode, im.tostring())
        bitmap_bw = bitmap.bitmap(i+1, 0, image_bw, height=0.8)
        c.insert(bitmap_bw)

    grad = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        grad[i] = matrix[i]/np.sum(matrix[i])

    for i in range(nci):
        for j in range(ncj):
            p = path.rect(i+1, j+1, 1, 1)
            c.stroke(p, [deco.filled([gray(1-grad[i, j])])])
            c.text(i+1, j+1, str(int(matrix[i,j])), [text.size(-1), rgb.blue])
    c.text(-1, (nci/2)+1, sensori, [text.size(-1)])
    c.text((ncj/2)+1, -1, sensorj, [text.size(-1)])
    # Margings for better printing
    c.text(-1.5, (nci/2)+1, " ", [text.size(-1)])
    c.text(nci+2, (nci/2)+1, " ", [text.size(-1)])
    c.text((ncj/2)+1, -1.5, " ", [text.size(-1)])
    c.text((ncj/2)+1, ncj+2,  " ", [text.size(-1)])


def compute_data_labels(fname, dfilec, dfile, sensorref, sensor):
    """
    Computes the labels of the data using the centroids of the cluster in the file
    the labels are relabeled acording to the matching with the reference sensor
    :param dfile:
    :param sensor:
    :return:
    """
    f = h5py.File(fname + '.hdf5', 'r')

    d = f[dfilec + '/' + sensor + '/Clustering/' + 'Centers']
    centers = d[()]
    d = f[dfile + '/' + sensor + '/' + 'PeaksResamplePCA']
    data = d[()]
    d = f[dfilec + '/' + sensorref + '/Clustering/' + 'Centers']
    centersref = d[()]
    f.close()

    # clabels, _ = pairwise_distances_argmin_min(centers, centersref)
    #
    # m = Munkres()
    # dist = euclidean_distances(centers, centersref)
    # indexes = m.compute(dist)
    # print indexes
    # print clabels
    labels, _ = pairwise_distances_argmin_min(data, centers)
    return labels #[indexes[i][1] for i in labels]

if __name__ == '__main__':

    window = 400
    print 'W=', int(round(window))
    # 'e150514''e120503''e110616''e150707''e151126''e120511'
    lexperiments = ['e120511e']

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()
    if args.exp:
        lexperiments = args.exp


    peakdata = {}
    for expname in lexperiments:
        datainfo = experiments[expname]

        # dfile = datainfo.datafiles[0]
        for dfile in datainfo.datafiles:
            print dfile

            lsens_labels = []
            #compute the labels of the data
            for sensor in datainfo.sensors:
                lsens_labels.append(compute_data_labels(datainfo.dpath + '/' + datainfo.name + '/' + datainfo.name,
                                                        datainfo.datafiles[0], dfile, datainfo.sensors[0], sensor))

            # Times of the peaks
            ltimes = []
            expcounts = []
            f = h5py.File(datainfo.dpath + '/' + datainfo.name + '/' + datainfo.name + '.hdf5', 'r')
            for sensor in datainfo.sensors:
                d = f[dfile + '/' + sensor + '/' + 'Time']
                data = d[()]
                expcounts.append(data.shape[0])
                ltimes.append(data)
            f.close()

            lsynchs = compute_synchs(ltimes, lsens_labels, window=window)

            contingency = peaks_contingency(lsynchs, datainfo.sensors, datainfo.clusters)

            # print len(contingency)
            # for ci in contingency:
            #     print len(ci)
            #     # for cj in ci:
            #     #     print cj

            pages = []
            for i, (sensori, nci) in enumerate(zip(datainfo.sensors, datainfo.clusters)):
                for j, (sensorj, ncj) in enumerate(zip(datainfo.sensors, datainfo.clusters)):
                    if i != j:
                        c = canvas.canvas()
                        draw_page(c, expname, sensori, sensorj, nci, ncj, contingency[i][j])
                        pages.append(document.page(c))

            d = document.document(pages)

            d.writePDFfile(datainfo.dpath + '/' + datainfo.name + '/Results/' + dfile + '-Contingency-' + str(int(window/10)) + '.pdf')
