"""
.. module:: PeaksSynchro

PeaksSynchro
*************

:Description: PeaksSynchro

    Analysis of syncronization patterns

    draw_synchs - Generates a file for each part of the experiment representing the synchronizations through time
    length_synch_frequency_histograms -
                        - Genrates a file for each part of the experiment with the histogram of the lengths of the
                          syncrhonizations
:Authors: bejar
    

:Version: 

:Created on: 19/11/2014 10:52 

"""

__author__ = 'bejar'

import pylab as P
from pylab import *
from pyx import *
from pyx.color import cmyk, rgb
from util.plots import plotMatrices, plotMatrix
from util.misc import normalize_matrix, compute_frequency_remap
import h5py
import numpy as np
from munkres import Munkres
from Config.experiments import experiments
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
import argparse
import cairo as cr

voc = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'

coormap = {'L4ci': (1, 1),
           'L4cd': (1, 2),
           'L5ri': (2, 1),
           'L5rd': (2, 2),
           'L5ci': (3, 1),
           'L5cd': (3, 2),
           'L6rd': (4, 1),
           'L6ri': (4, 2),
           'L6ci': (5, 1),
           'L6cd': (5, 2),
           'L7ri': (6, 1),
           'L7rd': (6, 2)
           }

def compute_RGB(val):
    """
    Return a rgb triple from its integer value

    :param val:
    :return:
    """
    b = val % 256
    val = int(val / 256)
    g = val % 256
    r = int(val / 256)
    return r/256.0, g/256.0, b/256.0

def choose_color(nsym):
    """
    selects the  RBG colors from a range with maximum nsym colors
    :param mx:
    :return:
    """
    cols = [(1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0), (1.0,1.0,0.0), (1.0,0.0,1.0), (0.0,1.0,1.0)]
    rep = nsym/len(cols)
    if nsym % len(cols) != 0:
        rep += 1
    fr = np.arange(1.0,0.0,-1.0/rep)
    lcols = []
    for i in fr:
        for c in cols:
            lcols.append(rgb(c[0]*i, c[1]*i, c[2]*i))
    return lcols

# def choose_color(nsym):
#     """
#     selects the  RBG colors at random
#     :param mx:
#     :return:
#     """
#     lcols = []
#     for c in np.random.rand(nsym,3):
#         lcols.append(rgb(c[0], c[1], c[2]))
#     return lcols


def gen_data_matrix(lines, clusters):
    """
    Generates a datastructure to store the peaks coincidences

    :param lines:
    :return:
    """

    mtrx = []
    for i, nci in zip(range(len(lines)), clusters):
        imtrx = []
        for j, ncj in zip(range(len(lines)), clusters):
            if i != j:
                imtrx.append(np.zeros((nci, ncj)))
            else:
                imtrx.append(None)
        mtrx.append(imtrx)
    return mtrx


def gen_peaks_contingency(peakdata, sensors, dfile, clusters):
    """
    Generates PDFs with the association frequencies of the peaks
    Each sensor with the synchronizations to the other sensors desagregated by peak class

    :return:
    """
    pk = peakdata

    dmatrix = gen_data_matrix(sensors, clusters)

    for p in pk:
        # print p
        for ei in p:
            for ej in p:
                if ei[0] != ej[0]:
                    #print ei[0], ej[0], ei[1][0], ej[1][0]
                    m = dmatrix[ei[0]][ej[0]]
                    m[ei[2]][ej[2]] += 1
                    # m = dmatrix[ej[0]][ei[0]]
                    # m[ej[1][0]][ei[1][0]] += 1
                    #

    for ln in range(len(sensors)):
        mt = dmatrix[ln]
        lplot = []

        for i in range(len(sensors)):
            if i != ln:
                #print normalize_matrix(mt[i]), sensors[ln], sensors[i]
                lplot.append((normalize_matrix(mt[i]), sensors[i]))
        plotMatrices(lplot, 6, 2, 'msynch-' + datainfo.name + '-' + dfile + '-' + sensors[ln], sensors[ln],
                     datainfo.dpath + '/' + datainfo.name + '/' + '/Results/')



def lines_coincidence_matrix(peaksynchs, sensors):
    """
    Computes a contingency matrix of how many times two lines have been synchronized

    :param peaksynchs:
    :return:
    """
    coinc = np.zeros((len(sensors), len(sensors)))

    for syn in peaksynchs:
        for i in syn:
            for j in syn:
                if i[0] != j[0]:
                    coinc[i[0], j[0]] += 1

    return coinc


def coincidence_contingency(peaksynchs, dfile, sensors):
    """
    Computes the contingency matrix of the proportion of syncronizations for each sensor over the total

    :param peaksyhchs:
    :param sensors:
    :return:
    """

    cmatrix = lines_coincidence_matrix(peaksynchs, sensors)
    cmatrix /= len(peaksynchs)
    sns.heatmap(cmatrix, annot=True, fmt="2.2f", cmap="afmhot_r", xticklabels=sensors, yticklabels=sensors, vmin=0, vmax=0.6)

    plt.title(dfile, fontsize=48)
    plt.savefig(datainfo.dpath + '/' + datainfo.name + '/' + '/Results/' + dfile + '-psync-corr.pdf',
                orientation='landscape', format='pdf')

    plt.close()


def synch_coincidence_matrix(peaksynchs, exp, sensors, expcounts, window):
    """
    Computes the probability of association among the peaks from the different sensors

    :param peaksynchs:
    :param expcounts:
    :return:
    """
    cmatrix = lines_coincidence_matrix(peaksynchs, sensors)
    corrmatrix = np.zeros((len(sensors), len(sensors)))
    for i in range(len(sensors)):
        for j in range(len(sensors)):
            if i != j:
                cab = cmatrix[i, j]
                ca = expcounts[i]
                cb = expcounts[j]
                tot = (ca + cb - cab) * 1.0
                corrmatrix[i, j] = cab / tot
    corrmatrix[0, 0] = 1.0
    plotMatrix(corrmatrix, exp + '-W' + str(int(window * 0.1)) + 'ms', exp + '-W' + str(int(window * 0.1)) + 'ms',
               [x for x in range(len(sensors))], [x for x in sensors], datainfo.dpath + '/Results/')



def length_synch_frequency_histograms(dsynchs, dfile, window):
    """
    Histograms of the frequencies of the lengths of the synchronizations
    :param dsynch: Dictionary with the synchronizations computed by

    :return:
    """
    x = []
    for pk in dsynchs:
        x.append(len(pk))

    P.figure()
    n, bins, patches = P.hist(x, max(x) - 1, normed=1, histtype='bar', fill=True)
    P.title('%s-W%d' % ( dfile, window), fontsize=48)
    P.savefig(datainfo.dpath + '/' + datainfo.name + '/' + '/Results/histo-' + datainfo.name + '-' + dfile + '-W' + str(window) + '.pdf', orientation='landscape', format='pdf')
    P.close()



def draw_synchs(peakdata, exp, sensors, window, nsym):
    """
    Generates a PDF of the synchronizations
    :param peakdata: Dictionary with the synchronizations computed by compute_syncs
    :param exps: Experiments in the dictionary
    :param window: Window used to determine the synchronizations
    :return:
    """
    def syncIcon(x, y, sensors, coord, lv, lcol, scale, can):
        scale *= 0.6
        y = (y / 2.5) + 2
        x += 5

        for al, v, col in zip(sensors, lv, lcol):
            c = coord[al]
            p = path.rect(y + (c[0] * scale), x + (c[1] * scale), scale, scale)
            if v:
                can.stroke(p, [deco.filled([col])])
            else:
                can.stroke(p)

 #   collist = [cmyk.GreenYellow, cmyk.Orange, cmyk.Mahogany, cmyk.OrangeRed, cmyk.Salmon, cmyk.Fuchsia, cmyk.Violet,
 #              cmyk.NavyBlue, cmyk.Cyan, cmyk.Emerald, cmyk.LimeGreen, cmyk.Sepia, cmyk.Tan]
    # Generates the list of colors
    collist = choose_color(nsym)

    ncol = 0
    npage = 1
    lpages = []
    pk = peakdata
    c = canvas.canvas()
    p = path.line(1, 0, 1, 28)
    c.stroke(p)
    c.text(1.5, 27, '%s - page: %d' % (exp, npage), [text.size(-1)])
    vprev = 0
    y = 0
    yt = 1.25
    for i, col in enumerate(collist):
        p = path.rect(0, i * 0.25, 0.25, 0.25)
        c.stroke(p, [deco.filled([col])])
    for i in range(len(pk)):

        l = [False] * len(sensors)
        lcol = [cmyk.White] * len(sensors)
        for p in pk[i]:
            l[p[0]] = True
            lcol[p[0]] = collist[p[2]]

        v = np.min([t for _, t, _ in pk[i]]) % 10000  # Time in a page (5000)
        if v < vprev:
            p = path.line(yt + 2.5, 0, yt + 2.5, 28)
            c.stroke(p)
            y += 6.3
            yt += 2.5
            ncol += 1
            if ncol % 7 == 0:
                # p=path.line(yt+2.5, 0, yt+2.5, 28)
                # c.stroke(p)
                npage += 1
                ncol = 0
                lpages.append(document.page(c))
                c = canvas.canvas()
                p = path.line(1, 0, 1, 28)
                c.stroke(p)
                c.text(1.5, 27, '%s - page: %d' % (exp, npage), [text.size(-1)])
                vprev = 0
                y = 0
                yt = 1.25

        vprev = v
        d = v - 1500  # end of the page (800)

        # proportion of x axis (200)
        c.text(yt, (d / 400.0) + 5.25, '%8s' % str(int(round(pk[i][0][1] * 0.6))), [text.size(-4)])
        syncIcon(d / 400.0, y, sensors, coormap, l, lcol, 0.25, c)

    # p=path.line(yt+2.5, 0, yt+2.5, 28)
    # c.stroke(p)

    d = document.document(lpages)

    d.writePDFfile(datainfo.dpath + '/' + datainfo.name + "/Results/peaksynchs-%s-%s-W%d" % (datainfo.name, exp, window))


def compute_synchs(seq, labels, window=15):
    """
    Computes the synchronizations of the peaks of several sensors

    :param seq: List of the peaks for all the sensors
                The list contains the time where the maximum of the peaks ocurr
    :param window: Window to consider that a set of peaks is synchronized
    :return: List of synchronizations (sensor, time, class)
    """

    def minind():
        """
        computes the index of the sequence with the lower value
        """
        mind = 0
        mval = float('inf')
        for i in range(len(seq)):
            if (len(seq[i]) > counts[i]) and (seq[i][counts[i]] < mval):
                mind = i
                mval = seq[i][counts[i]]
        return mind

    # Contadores de avance
    counts = [0] * len(seq)

    fin = False
    lsynch = []
    while not fin:
        # Compute the peak with the lower value
        imin = minind()
        if len(seq[imin]) > counts[imin] + 1 and \
            seq[imin][counts[imin]] <= (seq[imin][counts[imin] + 1] + window):
            # Look for the peaks inside the window length
            psynch = [(imin, seq[imin][counts[imin]], labels[imin][counts[imin]])]
            for i in range(len(seq)):
                if (len(seq[i]) > counts[i]) and (i != imin):
                    if seq[i][counts[i]] <= (seq[imin][counts[imin]] + window):
                        psynch.append((i, seq[i][counts[i]], labels[i][counts[i]]))
                        counts[i] += 1
            counts[imin] += 1
            if len(psynch) > 1:
                lsynch.append(psynch)
            else:
                lsynch.append(psynch)

                # print psynch
        else:
            counts[imin] += 1

        # We finish when all the counters have passed the lengths of the sequences
        fin = True
        for i in range(len(seq)):
            fin = fin and (counts[i] == len(seq[i]))
    return lsynch


def compute_data_labels(fname, dfilec, dfile, sensorref, sensor):
    """
    Computes the labels of the data using the centroids of the cluster in the file
    the labels are relabeled acording to the matching with the reference sensor

    Disabled the association using the Hungarian algorithm so the cluster index are
    the original ones

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

def select_sensor(synchs, sensor, slength):
    """
    Maintains only the syncs corresponding to the given sensor

    :param synchs:
    :param sensor:
    :return:
    """
    lres = []
    for syn in synchs:
        for s, _,_ in syn:
            if s == sensor and len(syn) >= slength:
                lres.append(syn)
    return lres

def save_sync_sequence(lsync, nfile):
    """
    Saves the list of synchronizations in a file with the number of sync signals

    :param lsync:
    :param file:
    :return:
    """
    ttable = '0123456789ABC'
    rfile = open(datainfo.dpath + '/' + datainfo.name + '/Results/' + 'seq-' + nfile + '.csv', 'w')
    for syn in lsync:
        mtime = min([v for _, v, _ in syn])
        rfile.write('%d, %s \n'% (mtime, ttable[len(syn)]))

    rfile.close()

# --------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    window = 2000

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
        for dfile in [datainfo.datafiles[0]]:
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

            #save_sync_sequence(lsynchs, dfile)

            # print len(lsynchs)
            # for i, s in enumerate(datainfo.sensors):
            #     lsyn_fil = select_sensor(lsynchs, i, 1)
            #     print s, len(lsyn_fil)

            peakdata = lsynchs
            #print peakdata
            #gen_peaks_contingency(peakdata, datainfo.sensors, dfile, datainfo.clusters)
            draw_synchs(peakdata, dfile, datainfo.sensors, window, datainfo.clusters[0])
            #length_synch_frequency_histograms(peakdata, dfile, window=int(round(window)))
            #synch_coincidence_matrix(peakdata, dfile, datainfo.sensors, expcounts, window)
            #coincidence_contingency(peakdata, dfile, datainfo.sensors)

