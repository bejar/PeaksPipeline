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
from Config.experiments import experiments
from sklearn.metrics import pairwise_distances_argmin_min
import seaborn as sns
import argparse
from util.misc import choose_color
from Matching.Match import compute_matching_mapping, compute_signals_matching
from fim import fpgrowth

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
                    # print ei[0], ej[0], ei[1][0], ej[1][0]
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
                # print normalize_matrix(mt[i]), sensors[ln], sensors[i]
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
    sns.heatmap(cmatrix, annot=True, fmt="2.2f", cmap="afmhot_r", xticklabels=sensors, yticklabels=sensors, vmin=0,
                vmax=0.6)

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
    P.title('%s-W%d' % (dfile, window), fontsize=48)
    P.savefig(datainfo.dpath + '/' + datainfo.name + '/' + '/Results/histo-' + datainfo.name + '-' + dfile + '-W' + str(
        window) + '.pdf', orientation='landscape', format='pdf')
    P.close()


def draw_synchs(peakdata, exp, ename, sensors, window, nsym, lmatch=0, dmappings=None):
    """
    Generates a PDF of the synchronizations
    :param peakdata: Dictionary with the synchronizations computed by compute_syncs
    :param exps: Experiments in the dictionary
    :param window: Window used to determine the synchronizations
    :return:
    """

    def syncIcon(x, y, sensors, coord, lv, lcol, scale, can):
        """
        Generates the drawing for a syncronization

        :param x:
        :param y:
        :param sensors:
        :param coord:
        :param lv:
        :param lcol:
        :param scale:
        :param can:
        :param dmappings:
        :return:
        """
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

    # Generates the list of colors
    if lmatch == 0:
        collist = choose_color(nsym)
    else:
        collist = choose_color(lmatch)

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

    # Paleta de colores
    for i, col in enumerate(collist):
        p = path.rect(0, i * 0.25, 0.25, 0.25)
        c.stroke(p, [deco.filled([col])])

    for i in range(len(pk)):
        l = [False] * len(sensors)
        lcol = [cmyk.White] * len(sensors)
        for p in pk[i]:
            l[p[0]] = True
            sns = sensors[p[0]]
            if dmappings is not None:
                lcol[p[0]] = collist[int(dmappings[sns][p[2]])]
            else:
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

    d.writePDFfile(
        datainfo.dpath + '/' + datainfo.name + "/Results/peaksynchs-%s-%s-%s-W%d" % (datainfo.name, exp, ename, window))


def draw_synchs_boxes(pk, exp, ename, sensors, window, nsym, lmatch=0, dmappings=None):
    """
    Draws syncronizations and their time window boxes

    :param peakdata:
    :param exp:
    :param ename:
    :param sensors:
    :param window:
    :param nsym:
    :param lmatch:
    :param dmappings:
    :return:
    """
    if lmatch != 0:
        collist = choose_color(lmatch)
    else:
        collist = choose_color(nsym)

    lpages = []

    c = canvas.canvas()

    # Paleta de colores
    for i, col in enumerate(collist):
        p = path.rect((i * 0.25) + 5, 7, 0.25, 0.25)
        c.stroke(p, [deco.filled([col])])

    if lmatch != 0:
        dpos = -len(lmatch)-1
    else:
        dpos = -nsym -1

    step = 200
    tres = 500.0
    stt = -1

    c.text(0, 10, "%s-%s-%s" % (datainfo.name, dfile, ename), [text.size(5)])

    for i in range(len(pk)):
        l = [False] * len(sensors)
        lcol = [cmyk.White] * len(sensors)
        ptime = np.zeros(len(sensors))
        for p in pk[i]:
            sns = sensors[p[0]]
            l[p[0]] = True
            ptime[p[0]] = p[1]
            if dmappings is not None:
                lcol[p[0]] = collist[int(dmappings[sns][p[2]])]
            else:
                lcol[p[0]] = collist[p[2]]

        tm1 = np.min([t for _, t, _ in pk[i]])
        tm2 = np.max([t for _, t, _ in pk[i]])
        coord = tm1 / tres
        lng = (tm2 - tm1) / tres

        dcoord = int(coord / step)
        ypos = dcoord * dpos
        if stt != dcoord:
            stt = dcoord
            c.text(-2, 3 + ypos, str(tm1), [text.size(3)])
            for j, sn in enumerate(sensors):
                c.text(1, 3 + ypos + 1 - j, sn, [text.size(3)])

        p = path.rect(3 + (coord - (dcoord * step)), 3 + ypos + 2 - len(sensors), lng + .05, len(sensors))
        c.stroke(p)

        for j in range(len(sensors)):

            if l[j]:
                tcoord = ptime[j] / tres
                c.fill(path.rect(3 + (tcoord - (dcoord * step)), 3 + ypos + 1 - j, 0.05, 1), [lcol[j]])
                # p = path.line(3 + (tcoord - (dcoord*step)), 3+ypos + 1+j,  3 + (tcoord - (dcoord*step)), 3+ypos + 2+j)
                # c.stroke(p, [deco.filled([lcol[j]])])

    lpages.append(document.page(c))

    d = document.document(lpages)

    d.writePDFfile(
        datainfo.dpath + '/' + datainfo.name + "/Results/peakssyncboxes-%s-%s-%s" % (datainfo.name, dfile, ename))


def compute_synchs(seq, labels, window=15, minlen=1):
    """
    Computes the synchronizations of the peaks of several sensors

    :param seq: List of the peaks for all the sensors
                The list contains the time where the maximum of the peaks ocurr
    :param labels: Labels of the classes of the peaks
    :param window: Window to consider that a set of peaks is synchronized
    :param minlen: Minimum length of the syncronization
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
            if len(psynch) >= minlen:
                lsynch.append(psynch)
                # else:
                #     lsynch.append(psynch)

                # print psynch
        else:
            counts[imin] += 1

        # We finish when all the counters have passed the lengths of the sequences
        fin = True
        for i in range(len(seq)):
            fin = fin and (counts[i] == len(seq[i]))
    return lsynch


def compute_data_labels(fname, dfilec, dfile, sensor):
    """
    Computes the labels of the data using the centroids of the cluster in the file
    the labels are relabeled acording to the matching with the reference sensor

    Disabled the association using the Hungarian algorithm so the cluster index are
    the original ones

    :param dfile:
    :param sensor:
    :return:
    """
    f = h5py.File(datainfo.dpath + '/' + fname + '/' + fname + '.hdf5', 'r')

    d = f[dfilec + '/' + sensor + '/Clustering/' + 'Centers']
    centers = d[()]
    d = f[dfile + '/' + sensor + '/' + 'PeaksResamplePCA']
    data = d[()]
    f.close()

    labels, _ = pairwise_distances_argmin_min(data, centers)
    return labels


def select_sensor(synchs, sensor, slength):
    """
    Maintains only the syncs corresponding to the given sensor

    :param synchs:
    :param sensor:
    :return:
    """
    lres = []
    for syn in synchs:
        for s, _, _ in syn:
            if s == sensor and len(syn) >= slength:
                lres.append(syn)
    return lres


def save_sync_sequence(lsync, nfile, lengths=False):
    """
    Saves the list of synchronizations in a file with the number of sync signals

    if lengths is true ony saves the length of the syncronization and the time of the first peak

    :param lsync:
    :param file:
    :return:
    """
    ttable = '0123456789ABC'
    rfile = open(datainfo.dpath + '/' + datainfo.name + '/Results/' + 'seq-' + nfile + '.csv', 'w')

    if lengths:
        for syn in lsync:
            mtime = min([v for _, v, _ in syn])
            rfile.write('%d, %s \n' % (mtime, ttable[len(syn)]))
    else:
        for syn in lsync:
            rfile.write('%s\n' % syn)

    rfile.close()

def compute_frequent_transactions(lsynchs, sup, lsensors):
    """
    Applies FP-growth for finding the frequent transactions in the syncronizations

    :return:
    """
    ltrans = []

    for synch in lsynchs:
        trans = []
        for sn, _, cl in synch:
            trans.append('%s-C%s'%(lsensors[sn],str(cl)))
        ltrans.append(trans)

    lfreq = []
    cnt_len = np.zeros(len(lsensors))
    for itemset, sval in fpgrowth(ltrans, supp=-sup, zmin=2, target='m'):
        lfreq.append((itemset, sval))
        cnt_len[len(itemset)-2] += 1

    return lfreq, cnt_len



# --------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    window = 400

    print 'W=', int(round(window))
    # 'e120503''e110616''e150707''e151126''e120511''e110906e''e150514'
    lexperiments = ['e110906e']

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")
    parser.add_argument('--save', help="Save Synchronizations", action='store_true', default=False)
    parser.add_argument('--histogram', help="Save length histograms", action='store_true', default=False)
    parser.add_argument('--coincidence', help="Save coincidence matrix", action='store_true', default=False)
    parser.add_argument('--contingency', help="Save peaks contingency matrix", action='store_true', default=False)
    parser.add_argument('--matching', help="Perform matching of the peaks", action='store_true', default=True)
    parser.add_argument('--boxes', help="Draws the syncronization in grouping boxes", action='store_true', default=True)
    parser.add_argument('--draw', help="Draws the syncronization matching", action='store_true', default=True)
    parser.add_argument('--rescale', help="Rescale the peaks for matching", action='store_true', default=False)
    parser.add_argument('--frequent', help="Computes frequent transactions algorithm for the synchonization", action='store_true', default=True)

    args = parser.parse_args()
    if args.exp:
        lexperiments = args.exp

    args.matching = True
    args.histogram = True
    args.draw = True
    args.boxes = False
    args.rescale = True

    # Matching parameters
    isig = 2
    fsig = 10

    peakdata = {}
    for expname in lexperiments:

        datainfo = experiments[expname]

        # dfile = datainfo.datafiles[0]
        for dfile, ename in zip(datainfo.datafiles, datainfo.expnames):
            print dfile

            lsens_labels = []
            if args.matching:
                lsensors = datainfo.sensors[isig:fsig]
                lclusters = datainfo.clusters[isig:fsig]
                smatching = compute_signals_matching(expname, lsensors, rescale=args.rescale)
            else:
                lsensors = datainfo.sensors
                lclusters = datainfo.clusters
                smatching = []

            # compute the labels of the data
            for sensor in lsensors:
                lsens_labels.append(compute_data_labels(datainfo.name, datainfo.datafiles[0], dfile, sensor))

            # Times of the peaks
            ltimes = []
            expcounts = []
            f = h5py.File(datainfo.dpath + '/' + datainfo.name + '/' + datainfo.name + '.hdf5', 'r')
            for sensor in lsensors:
                d = f[dfile + '/' + sensor + '/' + 'Time']
                data = d[()]
                expcounts.append(data.shape[0])
                ltimes.append(data)
            f.close()

            lsynchs = compute_synchs(ltimes, lsens_labels, window=window, minlen=1)

            if args.save:
                save_sync_sequence(lsynchs, dfile)

            # print len(lsynchs)
            # for i, s in enumerate(datainfo.sensors):
            #     lsyn_fil = select_sensor(lsynchs, i, 1)
            #     print s, len(lsyn_fil)

            # gen_peaks_contingency(peakdata, datainfo.sensors, dfile, datainfo.clusters)

            if args.matching:
                dmappings = {}
                for ncl, sensor in zip(lclusters, lsensors):
                    dmappings[sensor] = compute_matching_mapping(ncl, sensor, smatching)
            else:
                dmappings = None

            if args.draw and args.matching:

                draw_synchs(lsynchs, dfile, ename, lsensors, window, datainfo.clusters[0], lmatch=len(smatching),
                            dmappings=dmappings)

            if args.boxes:
                draw_synchs_boxes(lsynchs, dfile, ename, lsensors, window, datainfo.clusters[0], lmatch=len(smatching),
                                  dmappings=dmappings)

            if args.histogram:
                length_synch_frequency_histograms(lsynchs, dfile, window=int(round(window)))

            if args.coincidence:
                synch_coincidence_matrix(lsynchs, dfile, lsensors, expcounts, window)

            if args.contingency:
                coincidence_contingency(lsynchs, dfile, lsensors)

            if args.frequent:
                lfreq, cntlen = compute_frequent_transactions(lsynchs, sup=50, lsensors=lsensors)
                print len(lfreq), cntlen
