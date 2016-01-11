"""
.. module:: ComputeSubsequences

ComputeSubsequences
*************

:Description: ComputeSubsequences

    

:Authors: bejar
    

:Version: 

:Created on: 03/10/2014 9:55 

"""

__author__ = 'bejar'

import operator
import h5py
import numpy as np

from Config.experiments import experiments
import scipy.io
from pylab import *
import seaborn as sns

from Secuencias.rstr_max import *
from util.misc import compute_frequency_remap
from sklearn.metrics import pairwise_distances_argmin_min
import random
import string
import os

def randomize_string(s):
    l = list(s)
    random.shuffle(l)
    result = ''.join(l)
    return result

def drawgraph_alternative(nnodes, edges, nfile, sensor, dfile, legend, partition):
    rfile = open(datainfo.dpath + '/' + datainfo.name + '/Results/maxseqAlt-' + nfile + '-' + dfile + '-' + sensor + '.dot', 'w')

    rfile.write('digraph G {\nsize="20,20"\nlayout="neato"\n' +
                'imagepath="' + datainfo.dpath + '/'+ datainfo.name + '/Results/icons/' + '"\n' +
                'imagescale=true' + '\n' +
                'labeljust=r' + '\n' +
                'labelloc=b' + '\n' +
                'nodesep=0.4' + '\n' +
                'fontsize="30"\nlabel="' + legend + '"\n')

    radius = 5.0

    for i in range(nnodes):
        posx = -np.cos(((np.pi * 2) / nnodes) * i + (np.pi / 2)) * radius
        posy = np.sin(((np.pi * 2) / nnodes) * i + (np.pi / 2)) * radius
        rfile.write(str(i) + '[label="' + str(i + 1) + '",labeljust=l, labelloc=b, fontsize="24",height="0.2"' +
                    ', image="' + datainfo.name + sensor + '.cl' + str(i+1) + '.png' + '"' +
                    ', pos="' + str(posx) + ',' + str(posy) + '!", shape = "square"];\n')

    for e, nb, pe in edges:
        if len(e) == 2:
            rfile.write(str(e[0]) + '->' + str(e[1]))
            for lelem, color in partition:
                if e[0] in lelem:
                    rfile.write('[color="'+color+'"]')
            rfile.write('\n')

    rfile.write('}\n')

    rfile.close()
    os.system('dot -Tpdf '+datainfo.dpath + '/'+ datainfo.name+ '/Results/maxseqAlt-' + nfile + '-' + dfile + '-' + sensor + '.dot ' + '-o '
              + datainfo.dpath + '/'+ datainfo.name+ '/Results/maxseqAlt-' + nfile + '-' + dfile + '-' + sensor + '.pdf')
    os.system(' rm -fr ' + datainfo.dpath + '/'+ datainfo.name+ '/Results/maxseqAlt-' + nfile + '-' + dfile + '-' + sensor + '.dot')


def drawgraph(nnodes, edges, nfile, sensor, dfile, legend):
    rfile = open(datainfo.dpath + '/' + datainfo.name + '/Results/maxseq-' + nfile + '-' + dfile + '-' + sensor + '.dot', 'w')

    rfile.write('digraph G {\nsize="20,20"\nlayout="neato"\n' +
                'imagepath="' + datainfo.dpath + '/'+ datainfo.name + '/Results/icons/' + '"\n' +
                'imagescale=true' + '\n' +
                'labeljust=r' + '\n' +
                'labelloc=b' + '\n' +
                'nodesep=0.4' + '\n' +
                'fontsize="30"\nlabel="' + legend + '"\n')

    radius = 5.0

    for i in range(nnodes):
        posx = -np.cos(((np.pi * 2) / nnodes) * i + (np.pi / 2)) * radius
        posy = np.sin(((np.pi * 2) / nnodes) * i + (np.pi / 2)) * radius
        rfile.write(str(i) + '[label="' + str(i + 1) + '",labeljust=l, labelloc=b, fontsize="24",height="0.2"' +
                    ', image="' + datainfo.name + sensor + '.cl' + str(i+1) + '.png' + '"' +
                    ', pos="' + str(posx) + ',' + str(posy) + '!", shape = "square"];\n')

    for e, nb, pe in edges:
        if len(e) == 2:
            rfile.write(str(e[0]) + '->' + str(e[1]))
            if pe>= 0.014:
                rfile.write('[color="blue"]')
            if 0.014>pe> 0.007:
                rfile.write('[color="red"]')
            if pe < 0.007:
                rfile.write('[color="green"]')
            rfile.write('\n')

    rfile.write('}\n')

    rfile.close()
    os.system('dot -Tpdf '+datainfo.dpath + '/'+ datainfo.name+ '/Results/maxseq-' + nfile + '-' + dfile + '-' + sensor + '.dot ' + '-o '
              + datainfo.dpath + '/'+ datainfo.name+ '/Results/maxseq-' + nfile + '-' + dfile + '-' + sensor + '.pdf')
    os.system(' rm -fr ' + datainfo.dpath + '/'+ datainfo.name+ '/Results/maxseq-' + nfile + '-' + dfile + '-' + sensor + '.dot')


def drawgraph_with_edges(nnodes, edges, nfile, sensor):
    rfile = open(datainfo.dpath + '/'+ datainfo.name + '/Results/maxseq-' + nfile + '.dot', 'w')

    # rfile.write('digraph G {\nsize="6,6"\nlayout="neato"\nfontsize="30"\nlabel="'+nfile+'"\n')
    rfile.write('digraph G {\nsize="20,20"\nlayout="neato"\n' +
                'imagepath="' + datainfo.dpath + '/'+ datainfo.name+ '/Results/icons/' + '"\n' +
                'imagescale=true' + '\n' +
                'labeljust=r' + '\n' +
                'labelloc=b' + '\n' +
                'nodesep=0.4' + '\n' +
                'fontsize="30"\nlabel="' + nfile + '"\n')

    radius = 5.0

    for i in range(nnodes):
        posx = -np.cos(((np.pi * 2) / nnodes) * i + (np.pi / 2)) * radius
        posy = np.sin(((np.pi * 2) / nnodes) * i + (np.pi / 2)) * radius
        rfile.write(str(i) + '[label="' + str(i + 1) + '",labeljust=l, labelloc=b, fontsize="24",height="0.7"' +
                    ', image="' + sensor + '.cl' + str(i+1) + '.png' + '"' +
                    ', pos="' + str(posx) + ',' + str(posy) + '!", shape = "square"];\n')


        # rfile.write(str(i+1)+'[label="'+str(i+1) +
        #             '", fontsize="24",height="0.7"' +
        #             'image="'+rpath+'cl1.png'+'"' +
        #             ', pos="'+str(posx)+','+str(posy) +
        #             '!", shape = "circle"];\n')
    rfile.write('\n')
    for e in edges:
        if len(e) == 2:
            rfile.write(str(e[0]) + '->' + str(e[1]) + '\n')

    rfile.write('}\n')

    rfile.close()




def max_seq_long(nexp, clpeaks, timepeaks, sup, nfile, gap=0):
    """
    Secuencias con soporte mas que un limite grabadas en fichero de texto

    :param nexp:
    :param clpeaks:
    :param timepeaks:
    :param sup:
    :param nfile:
    :param remap:
    :param gap:
    :return:
    """
    # Select the index of the experiment
    peakini = 0
    i = 0
    while i < nexp:
        exp = timepeaks[i]
        peakini += exp.shape[0]
        i += 1

    exp = timepeaks[nexp]
    peakend = peakini + exp.shape[0]

    # Build the sequence string
    peakstr = ''

    for i in range(peakini, peakend):
        peakstr += voc[clpeaks[i][0]]
        if i < peakend - 1 and gap != 0:
            if (timepeaks[nexp][i - peakini + 1] - timepeaks[nexp][i - peakini]) > gap:
                peakstr += '#'

    # Compute the sufix array
    rstr = Rstr_max()
    rstr.add_str(peakstr)

    r = rstr.go()

    # Compute the sequences that have minimum support
    lstrings = []

    for (offset_end, nb), (l, start_plage) in r.iteritems():
        ss = rstr.global_suffix[offset_end - l:offset_end]
        id_chaine = rstr.idxString[offset_end - 1]
        s = rstr.array_str[id_chaine]
        # print '[%s] %d'%(ss.encode('utf-8'), nb)
        if nb > sup and len(ss.encode('utf-8')) > 1:
            lstrings.append((ss.encode('utf-8'), nb))

    lstrings = sorted(lstrings, key=operator.itemgetter(0), reverse=True)
    lstringsg = []
    rfile = open(datainfo.dpath + '/'+ datainfo.name + '/Results/maxseqlong-' + nfile + '.txt', 'w')
    cntlong = np.zeros(10)
    for seq, s in lstrings:
        wstr = ''
        if not '#' in seq:
            sigsym = []
            for c in range(len(seq)):
                wstr += '{0:0>2d}'.format(voc.find(seq[c]))
                rmp = voc.find(seq[c])
                sigsym.append(rmp)
                if c < (len(seq) - 1):
                    wstr += ' - '
            lstringsg.append(sigsym)
            wstr += ' = ' + str(s)

            rfile.write('[' + str(len(seq)) + '] ' + wstr + '\n')
            cntlong[len(seq)] += 1
    rfile.close()
    for i in range(2, 10):
        print(i, ':', cntlong[i])
    print('----------')


def max_seq_exp(nfile, clpeaks, timepeaks, sensor, dfile, ename, nclust, gap=0, sup=None, rand=False, galt=False, partition=None):
    """
    Generates frequent subsequences and the graphs representing the two step frequent
    subsequences

    Auto tunes the minimum support

    :param nexp:
    :param clpeaks:
    :param timepeaks:
    :param sup:
    :param nfile:
    :param remap:
    :param gap:
    :return:
    """
    # Build the sequence string
    peakstr = ''

    peakfreq = {'#': 0}

    for i in range(timepeaks.shape[0]):
        peakstr += voc[clpeaks[i]]
        if i < timepeaks.shape[0] - 1 and gap != 0:
            if (timepeaks[i + 1] - timepeaks[i]) > gap:
                peakstr += '#'
                peakfreq['#'] += 1

        if voc[clpeaks[i]] in peakfreq:
            peakfreq[voc[clpeaks[i]]] += 1
        else:
            peakfreq[voc[clpeaks[i]]] = 1

    if rand:
        peakstr = randomize_string(peakstr)
    # print peakend - peakini, len(peakstr), len(peakstr)*(1.0 / (len(peakfreq)*len(peakfreq)))
    # Support computed heuristcally

    if sup is None:
        sup = int(round(len(peakstr) * (1.0 / (len(peakfreq) * len(peakfreq)))) * 1.0)
    print(sup)

    for l in peakfreq:
        peakfreq[l] = (peakfreq[l] * 1.0) / len(peakstr)

    # Compute the sufix array
    rstr = Rstr_max()
    rstr.add_str(peakstr)

    r = rstr.go()

    # Compute the sequences that have minimum support
    lstrings = []

    for (offset_end, nb), (l, start_plage) in r.iteritems():
        ss = rstr.global_suffix[offset_end - l:offset_end]
        id_chaine = rstr.idxString[offset_end - 1]
        s = rstr.array_str[id_chaine]
        if nb > sup and len(ss.encode('utf-8')) > 1:
            lstrings.append((ss.encode('utf-8'), nb))

    lstrings = sorted(lstrings, key=operator.itemgetter(0), reverse=True)
    lstringsg = []
    randname = ''
    if rand:
        randname = randname.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

    rfile = open(datainfo.dpath+ '/'+ datainfo.name + '/Results/maxseq-' + nfile + '-' + ename + '-' + sensor + '-' + randname + '.txt', 'w')

    mfreq = np.zeros((nclust, nclust))
    for seq, s in lstrings:
        wstr = ''
        prob = 1.0
        if not '#' in seq:
            if len(seq) == 2:
                mfreq[voc.find(seq[0]),  voc.find(seq[1])] = int(s)

            sigsym = []
            for c in range(len(seq)):
                wstr += str(voc.find(seq[c]))
                rmp = voc.find(seq[c])
                sigsym.append(rmp)
                prob *= peakfreq[seq[c]]
                if c < (len(seq) - 1):
                    wstr += ' - '

            lstringsg.append((sigsym, prob, (s * 1.0) / (len(peakstr) - 1)))
            wstr += ' = ' + str(s) + ' ( ' + str(prob) + ' / ' + str((s * 1.0) / (len(peakstr) - 1)) + ' )'
            rfile.write(wstr + '\n')
    rfile.close()

    fig = plt.figure()
    sns.heatmap(mfreq, annot=True, fmt='.0f', cbar=False, xticklabels=range(1,nclust+1), yticklabels=range(1,nclust+1), square=True)
    plt.title(nfile + '-' + ename + '-' + sensor + ' sup(%d)' % sup)
    plt.savefig(datainfo.dpath + '/' + datainfo.name + '/Results/maxseq-histo' + datainfo.name + '-' + dfile + '-'
                + sensor  + '-freq.pdf', orientation='landscape', format='pdf')
    plt.close()

    nsig = len(peakfreq)
    if '#' in peakfreq:
        nsig -= 1

    if galt:
        drawgraph_alternative(nclust, lstringsg, nfile, sensor, dfile, nfile + '-' + ename + '-' + sensor + ' sup(%d)' % sup, partition=partition)
    else:
        drawgraph(nclust, lstringsg, nfile, sensor, dfile, nfile + '-' + ename + '-' + sensor + ' sup(%d)' % sup)


def max_peaks_edges(nexp, clpeaks, timepeaks, sup, gap=0):
    # Select the index of the experiment
    peakini = 0
    i = 0
    while i < nexp:
        exp = timepeaks[i]
        peakini += exp.shape[0]
        i += 1

    exp = timepeaks[nexp]
    peakend = peakini + exp.shape[0]

    # Build the sequence string
    peakstr = ''

    peakfreq = {'#': 0}

    for i in range(peakini, peakend):
        peakstr += voc[clpeaks[i][0]]
        if i < peakend - 1 and gap != 0:
            if (timepeaks[nexp][i - peakini + 1] - timepeaks[nexp][i - peakini]) > gap:
                peakstr += '#'
                peakfreq['#'] += 1

        if voc[clpeaks[i][0]] in peakfreq:
            peakfreq[voc[clpeaks[i][0]]] += 1
        else:
            peakfreq[voc[clpeaks[i][0]]] = 1

    print(peakend - peakini, len(peakstr), len(peakstr) * (1.0 / (len(peakfreq) * len(peakfreq))))

    for l in peakfreq:
        peakfreq[l] = (peakfreq[l] * 1.0) / len(peakstr)
    # print l, (peakfreq[l]* 1.0)/(peakend - peakini)
    #    print clpeaks[i][0]


    # Compute the sufix array
    rstr = Rstr_max()
    rstr.add_str(peakstr)

    r = rstr.go()

    # Compute the sequences that have minimum support
    lstrings = []

    for (offset_end, nb), (l, start_plage) in r.iteritems():
        ss = rstr.global_suffix[offset_end - l:offset_end]
        id_chaine = rstr.idxString[offset_end - 1]
        s = rstr.array_str[id_chaine]
        #print '[%s] %d'%(ss.encode('utf-8'), nb)
        if nb > sup and len(ss.encode('utf-8')) > 1:
            lstrings.append((ss.encode('utf-8'), nb))

    lstrings = sorted(lstrings, key=operator.itemgetter(0))
    lstringsg = []
    for seq, s in lstrings:
        wstr = ''
        prob = 1.0
        if not '#' in seq:
            sigsym = []
            for c in range(len(seq)):
                #rmp = remap[voc.find(seq[c])-1]
                rmp = voc.find(seq[c])
                sigsym.append(rmp)
                prob *= peakfreq[seq[c]]
            lstringsg.append(sigsym)

    return lstringsg

# ----------------------------------------


def generate_sequences(dfile, ename, timepeaks, clpeaks, sensor, ncl, gap, sup=None, rand=False, galt=False, partition=None):
    """
    Generates the frequent subsequences from the times of the peaks considering
    gap the minimum time between consecutive peaks that indicates a pause (time in the sampling frequency)

    :param dfile:
    :param timepeaks:
    :param clpeaks:
    :param sensor:
    :return:
    """
    max_seq_exp(datainfo.name, clpeaks, timepeaks, sensor, dfile, ename, ncl, gap=gap, sup=sup, rand=rand, galt=galt, partition=partition)


def generate_sequences_long(dfile, timepeaks, clpeaks, sensor, thres, gap):
    max_seq_long(exp, clpeaks, timepeaks, thres, sensor + '-' + dfile, gap=gap)


def generate_diff_sequences(dfile, timepeaks, clpeaks, sensor, gap):

    ledges1 = max_peaks_edges(1, clpeaks, timepeaks, 20, gap=gap)
    ledges2 = max_peaks_edges(2, clpeaks, timepeaks, 25, gap=gap)

    for e in ledges1:
        if e in ledges2:
            ledges2.remove(e)

    drawgraph_with_edges(ledges2, 'dif-' + sensor + '-' + 'ctrl2-capsa1')


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


voc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# line = 'L6ri'  # 'L6rd' 'L5ci' 'L6ri'
# clust = '.k15.n1'  # '.k20.n5' '.k16.n4' '.k15.n1'

# for line, clust, _ in aline:
#     print line
#     matpeaks = scipy.io.loadmat(datapath + 'Selected/centers.' + line + '.' + clust + '.mat')
#     mattime = scipy.io.loadmat(datapath + '/WholeTime.' + line + '.mat')
#
#     clpeaks = matpeaks['IDX']
#     timepeaks = mattime['temps'][0]
#
#     generate_sequences()

    #generate_sequences_long()

    #generate_sequences()

    #generate_diff_sequences()


# --------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # 'e110616''e120503''e150514' 'e150514''e150707'
    lexperiments = ['e151126']
    galt = True
    partition = [[[0, 1, 2, 3], 'red'], [[4, 5, 6, 7], 'blue'], [[8,9,10,11], 'green']]

    peakdata = {}
    for expname in lexperiments:

        datainfo = experiments[expname]
        f = h5py.File(datainfo.dpath + '/' + datainfo.name + '/' + datainfo.name + '.hdf5', 'r')

        for dfile, ename in zip(datainfo.datafiles, datainfo.expnames):
            print(dfile)

            for ncl, sensor in zip(datainfo.clusters, datainfo.sensors):
                if dfile + '/' + sensor + '/' + 'Time' in f:
                    clpeaks = compute_data_labels(datainfo.datafiles[0], dfile, sensor)
                    d = f[dfile + '/' + sensor + '/' + 'Time']
                    timepeaks = data = d[()]
                    generate_sequences(dfile, ename, timepeaks, clpeaks, sensor, ncl, gap=2000, sup=None, rand=False, galt=galt, partition=partition)
