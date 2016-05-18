"""
.. module:: ComputeSubsequences

ComputeSubsequences
*************

:Description: ComputeSubsequences

    Computes the frequent sequences for an experiments

    Generates circular graph for the sequences of length 2 and contingency tables for the counts of frequent sequencies
    of length 2

:Authors: bejar
    

:Version: 

:Created on: 03/10/2014 9:55 

"""


import h5py

from Config.experiments import experiments
from pylab import *
import seaborn as sns
from Secuencias.rstr_max import *
from sklearn.metrics import pairwise_distances_argmin_min
import random
import string
import os
import argparse
from pyx import *
import operator

from util.misc import choose_color
from Matching.Match import compute_matching_mapping, compute_signals_matching


__author__ = 'bejar'

def randomize_string(s):
    l = list(s)
    random.shuffle(l)
    result = ''.join(l)
    return result

def drawgraph_alternative(nnodes, edges, nfile, sensor, dfile, ename, legend, partition, lmatch=0, mapping=None, proportional=False, globalc=False):
    """
    Draws a circular graph of the frequent pairs coloring the edges according to the peaks partition

    :param nnodes:
    :param edges:
    :param nfile:
    :param sensor:
    :param dfile:
    :param ename:
    :param legend:
    :param partition:
    :param lmatch:
    :param mapping:
    :return:
    """

    ext = '' if lmatch == 0 else '-match'
    gclust = '' if not globalc else '.g'

    rfile = open(datainfo.dpath + '/' + datainfo.name + '/Results/maxseqAlt-' + nfile + '-' + dfile + '-' + sensor + '-' + ename + '-' + str(nnodes)  + ext + '.dot', 'w')

    rfile.write('digraph G {\nsize="20,20"\nlayout="neato"\n' +
                'imagepath="' + datainfo.dpath + '/'+ datainfo.name + '/Results/icons/' + '"\n' +
                'imagescale=true' + '\n' +
                'labeljust=r' + '\n' +
                'labelloc=b' + '\n' +
                'nodesep=0.4' + '\n' +
                'fontsize="30"\nlabel="' + legend + '"\n')

    radius = 5.0
    if lmatch == 0:
        roundlen = nnodes
    else:
        roundlen = lmatch

    for i in range(nnodes):
        if lmatch == 0:
            ind = i
        else:
            ind = mapping[i]

        posx = -np.cos(((np.pi * 2) / roundlen) * ind + (np.pi / 2)) * radius
        posy = np.sin(((np.pi * 2) / roundlen) * ind + (np.pi / 2)) * radius
        rfile.write(str(i) + '[label="' + str(i + 1) + '",labeljust=l, labelloc=b, fontsize="24",height="0.2"' +
                    ', image="' + datainfo.name + sensor + '.nc' + str(nnodes) + '.cl' + str(i+1) + gclust +'.png' + '"' +
                    ', pos="' + str(posx) + ',' + str(posy) + '!", shape = "square"];\n')
    amp = nnodes * 15
    for e, nb, pe in edges:
        if len(e) == 2:
            rfile.write(str(e[0]) + '->' + str(e[1]))
            if proportional:
                width = int(pe*amp)
                if width == 0:
                    width = 1
            else:
                width = 1
            for lelem, color in partition:
                if lmatch != 0:
                    if mapping[e[0]] in lelem:
                        rfile.write('[color="'+color+'" penwidth ="' + str(width) + '"]')
                else:
                    if e[0] in lelem:
                        rfile.write('[color="'+color+'" penwidth ="' + str(width) +'"]')

            rfile.write('\n')

    rfile.write('}\n')

    rfile.close()
    os.system('dot -Tpdf '+datainfo.dpath + '/'+ datainfo.name+ '/Results/maxseqAlt-' + nfile + '-' + dfile + '-' + sensor + '-' + ename + '-' + str(nnodes) + ext + '.dot ' + '-o '
              + datainfo.dpath + '/'+ datainfo.name+ '/Results/maxseqAlt-' + nfile + '-' + dfile + '-' + sensor + '-' + ename  + '-' + str(nnodes) + ext + '.pdf')
    os.system(' rm -fr ' + datainfo.dpath + '/'+ datainfo.name+ '/Results/maxseqAlt-' + nfile + '-' + dfile + '-' + sensor + '-' + ename  + '-' + str(nnodes) + ext + '.dot')


def drawgraph(nnodes, edges, nfile, sensor, dfile, legend, lmatch=0, mapping=None):
    """
    Draws a circular graph of the frequent pairs

    :param nnodes:
    :param edges:
    :param nfile:
    :param sensor:
    :param dfile:
    :param legend:
    :param lmatch:
    :param mapping:
    :return:
    """

    ext = '' if lmatch == 0 else '-match'
    rfile = open(datainfo.dpath + '/' + datainfo.name + '/Results/maxseq-' + nfile + '-' + dfile + '-' + sensor + '-' + str(nnodes) + ext + '.dot', 'w')

    rfile.write('digraph G {\nsize="20,20"\nlayout="neato"\n' +
                'imagepath="' + datainfo.dpath + '/'+ datainfo.name + '/Results/icons/' + '"\n' +
                'imagescale=true' + '\n' +
                'labeljust=r' + '\n' +
                'labelloc=b' + '\n' +
                'nodesep=0.4' + '\n' +
                'fontsize="30"\nlabel="' + legend + '"\n')

    radius = 5.0
    if lmatch == 0:
        roundlen = nnodes
    else:
        roundlen = lmatch

    for i in range(nnodes):
        if lmatch == 0:
            ind = i
        else:
            ind = mapping[i]
        posx = -np.cos(((np.pi * 2) / roundlen) * ind + (np.pi / 2)) * radius
        posy = np.sin(((np.pi * 2) / roundlen) * ind + (np.pi / 2)) * radius
        rfile.write(str(i) + '[label="' + str(i + 1) + '",labeljust=l, labelloc=b, fontsize="24",height="0.2"' +
                    ', image="' + datainfo.name + sensor + '.nc' + str(nnodes) + '.cl' + str(i+1) + '.png' + '"' +
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


    os.system('dot -Tpdf '+datainfo.dpath + '/'+ datainfo.name+ '/Results/maxseq-' + nfile + '-' + dfile + '-' + sensor + '-' + str(nnodes) + ext + '.dot ' + '-o '
              + datainfo.dpath + '/'+ datainfo.name+ '/Results/maxseq-' + nfile + '-' + dfile + '-' + sensor + '-' + str(nnodes) + ext + '.pdf')
    os.system(' rm -fr ' + datainfo.dpath + '/'+ datainfo.name+ '/Results/maxseq-' + nfile + '-' + dfile + '-' + sensor + '-' + str(nnodes) + ext + '.dot')


def drawgraph_with_edges(nnodes, edges, nfile, sensor):
    rfile = open(datainfo.dpath + '/'+ datainfo.name + '/Results/maxseq-' + nfile + '-' + str(nnodes) + '.dot', 'w')

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
                    ', image="' + sensor + '.nc' + str(nnodes) + '.cl' + str(i+1) + '.png' + '"' +
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


def freq_seq_positions(nfile, clpeaks, timepeaks, sensor, ename, nclust, gap=0, sup=None):
    """
    Generates a list of the positions of the frequent sequences

    :param nfile:
    :param clpeaks:
    :param timepeaks:
    :param sensor:
    :param dfile:
    :param ename:
    :param nclust:
    :param gap:
    :param sup:
    :return:
    """

    # First compute the sequences with a frequency larger than a threshold
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

    # Support computed heuristically
    if sup is None:
        sup = int(round(len(peakstr) * (1.0 / (len(peakfreq) * len(peakfreq)))) * 1.0)

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

    # Compute the sequences that do not include the pause and have length 2
    lseq = set()
    for seq, s in lstrings:
        if not '#' in seq and len(seq) == 2:
            lseq.add((voc.find(seq[0]), voc.find(seq[1])))

    lseqpos = []

    for i in range(timepeaks.shape[0]):
        if i < timepeaks.shape[0] - 1 and gap != 0:
            if (timepeaks[i + 1] - timepeaks[i]) < gap:
                if (clpeaks[i], clpeaks[i+1]) in lseq:
                    lseqpos.append([timepeaks[i],clpeaks[i], timepeaks[i+1],clpeaks[i+1]])
                else:
                    lseqpos.append([timepeaks[i],clpeaks[i]])
            else:
                lseqpos.append([timepeaks[i],clpeaks[i]])

    return lseqpos

def plot_sequences(nfile, lseq, nsym, sensor, lmatch=0, mapping=None):
    """
    plots the sequence of frequent pairs

    :param nfile:
    :param lseq:
    :return:
    """
    if lmatch != 0:
        collist = choose_color(lmatch)
    else:
        collist = choose_color(nsym)

    lpages = []

    c = canvas.canvas()

    if lmatch != 0:
        mtx = np.zeros(lmatch) + 0.5
        for i in range(nsym):
            mtx[mapping[i]] = 1
        for i, col in enumerate(collist):
            p = path.rect(3 + i , 7, 1,  mtx[i])
            c.stroke(p, [deco.filled([col])])
    else:
        for i, col in enumerate(collist):
            p = path.rect(3 + i , 7, 1,  1)
            c.stroke(p, [deco.filled([col])])

    dpos = -5
    step = 50
    tres = 5000.0
    stt = -1

    c.text(0, 10, "%s-%s-%s-%s" % (datainfo.name, dfile, sensor, ename), [text.size(5)])

    for seq in lseq:
        if len(seq) == 4:
            tm1, pk1, tm2, pk2 = seq
            coord = tm1 / tres
            lng = (tm2/tres) - coord
            if lmatch != 0:
                pk1 = int(mapping[pk1])
                pk2 = int(mapping[pk2])


            dcoord = int(coord/step)
            ypos = dcoord * dpos
            if stt != dcoord:
                stt = dcoord
                c.text(0, 3+ypos, str(tm1), [text.size(1)])
            p = path.rect(3+coord - (dcoord*step), 3+ypos, lng/2.0,  1)
            c.stroke(p, [deco.filled([collist[pk1]])])
            p = path.rect(3+coord - (dcoord*step) +(lng/2.0), 3+ypos, lng/2.0,  1)
            c.stroke(p, [deco.filled([collist[pk2]])])
            c.stroke(p)
            # p = path.line(3+coord - (dcoord*step), 3+ypos + 1,  3+coord - (dcoord*step), 3+ypos + 2)
            # c.stroke(p, [deco.filled([collist[pk1]])])
            c.fill(path.rect(3+coord - (dcoord*step), 3+ypos + 1, 0.05, 1),[collist[pk1]])
        else:
            tm1, pk1 = seq
            if lmatch != 0:
                pk1 = int(mapping[pk1])
            coord = tm1 / tres
            dcoord = int(coord /step)
            ypos = dcoord * dpos
            if stt != dcoord:
                stt = dcoord
                c.text(0,3+ ypos, str(tm1), [text.size(1)])
            # p = path.line(3+coord - (dcoord*step), 3+ypos + 1, 3+coord - (dcoord*step), 3+ypos + 2)
            # c.stroke(p, [deco.filled([collist[pk1]])])
            c.fill(path.rect(3+coord - (dcoord*step), 3+ypos + 1, 0.05, 1),[collist[pk1]])


    lpages.append(document.page(c))

    d = document.document(lpages)

    d.writePDFfile(datainfo.dpath + '/' + datainfo.name + "/Results/peaksseq-%s-%s-%s-%s-%d" % (datainfo.name, dfile, sensor, ename, nsym))


def peaks_sequence_frequent_strings(timepeaks, gap=0, rand=False, sup=None):
    """
    Computes the frequent strings in a string representing the sequence of peaks

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

    # Support computed heuristically
    if sup is None:
        sup = int(round(len(peakstr) * (1.0 / (len(peakfreq) * len(peakfreq)))) * 1.0)

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

    return peakstr, peakfreq, lstrings


def compute_pairs_distribution(peakstr, ncl):
    """
    Computes the distribution of the pairs in the sequence

    :param peakstr:
    :return:
    """

    symbols = [voc[i] for i in range(ncl)]
    symbols.append('#')

    print(len(symbols))
    dfreq = {}
    for i in symbols:
        dfreq[i] = {}
        for j in symbols:
            dfreq[i][j] = 0

    for i in range(len(peakstr)-1):
        dfreq[peakstr[i]][peakstr[i+1]] += 1

    lcounts = []

    for df in dfreq:
        for f in dfreq[df]:
            lcounts.append(dfreq[df][f])

    acounts = np.array(lcounts)/float((len(peakstr)-1))
    # fig = plt.figure()
    # #plt.plot(np.cumsum(sorted(acounts)))
    # #plt.plot(sorted(acounts))
    # plt.plot(acounts)
    # plt.show()
    return acounts



def save_frequent_sequences(nfile, peakstr, peakfreq, lstrings, sensor, dfile,  ename, nclust, lmatch=0, sup=None,
                            mapping=None, rand=False, galt=False, partition=None, save=(False, False, True),
                            proportional=False, globalc=False):
    """
    Saves a file with the frequent sequences for a file and sensor, a contingency table with the absolute frequency
    of the pairs and a circular graph with the pairs

    returns a list with the frequent strings and their probabilities
    :return:
    """
    # Support computed heuristically
    if sup is None:
        sup = int(round(len(peakstr) * (1.0 / (len(peakfreq) * len(peakfreq)))) * 1.0)

    lstringsg = []
    randname = ''
    if rand:
        randname = randname.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

    if lmatch == 0:
        mfreq = np.zeros((nclust, nclust))
    else:
        mfreq = np.zeros((lmatch, lmatch))

    fwstr = []
    for seq, s in lstrings:
        wstr = ''
        prob = 1.0
        if not '#' in seq:
            if len(seq) == 2:
                if lmatch == 0:
                    mfreq[voc.find(seq[0]),  voc.find(seq[1])] = int(s)
                else:
                    mfreq[mapping[voc.find(seq[0])],  mapping[voc.find(seq[1])]] = int(s)

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
            fwstr.append(wstr)

    ext = '' if lmatch == 0 else '-match'

    # List with the frequent strings and their probability (computed and theoretical)
    if save[0]:
        rfile = open(datainfo.dpath+ '/'+ datainfo.name + '/Results/maxseq-' + nfile + '-' + ename + '-' + sensor + '-' + str(nclust) + ext
                     + '-' + randname + '.txt', 'w')
        for line in fwstr:
            rfile.write(line + '\n')
        rfile.close()

    # Contingency table of the number of times a frequent sequence of length 2 has appeared
    if save[1]:
        fig = plt.figure()
        if lmatch != 0:
            sns.heatmap(mfreq, annot=True, fmt='.0f', cbar=False, xticklabels=range(1, lmatch+1), yticklabels=range(1, lmatch+1), square=True)
        else:
            sns.heatmap(mfreq, annot=True, fmt='.0f', cbar=False, xticklabels=range(1, nclust+1), yticklabels=range(1, nclust+1), square=True)

        plt.title(nfile + '-' + ename + '-' + sensor + ext + ' sup(%d)' % sup)
        plt.savefig(datainfo.dpath + '/' + datainfo.name + '/Results/maxseq-histo' + datainfo.name + '-' + dfile + '-'
                    + sensor + '-' + str(nclust) + ext + '-freq.pdf', orientation='landscape', format='pdf')
        plt.close()

    # Circular graph of the frequent sequences of length 2
    if save[2]:
        nsig = len(peakfreq)
        if '#' in peakfreq:
            nsig -= 1

        if galt:
            drawgraph_alternative(nclust, lstringsg, nfile, sensor, dfile, ename,
                                  nfile + '-' + ename + '-' + sensor + '-' + str(nclust) + ext + ' sup(%d)' % sup,
                                  partition=partition, lmatch=lmatch, mapping=mapping, proportional=proportional,
                                  globalc=globalc)

        else:
            drawgraph(nclust, lstringsg, nfile, sensor, dfile,
                      nfile + '-' + ename + '-' + sensor + '-' + str(nclust) + ext + ' sup(%d)' % sup,
                      lmatch=lmatch, mapping=mapping)

    return lstringsg


def sequence_to_string(nfile, clpeaks, timepeaks, sensor, ename, gap=0, npart=1, ncl=0):
    """
    Writes npart files with a strings representing the sequence of peaks.
    Each partition has a number of peaks proportional to the time and the gap is included a
    number of times proportional to the time distance among two peaks when it is larger than the gap value

    :param clpeaks:
    :param timepeaks:
    :param gap:
    :return:
    """
    # Build the sequence string

    mtime = timepeaks[-1]/npart
    peakstr = ''
    part = 1
    for i in range(timepeaks.shape[0]):
        if timepeaks[i] > part*mtime:
            rfile = open(datainfo.dpath + '/' + datainfo.name + '/Results/stringseq-%s%d-%s-%s-%d.txt'%(nfile, part, ename, sensor, ncl), 'w')
            for k in range(0, len(peakstr), 250):
                wstr = ''
                for j in range(250):
                    if k+j < len(peakstr):
                        wstr += peakstr[k+j]
                rfile.write(wstr + '\n')
            rfile.close()
            peakstr = ''
            part += 1
        peakstr += voc[clpeaks[i]]
        if i < timepeaks.shape[0] - 1 and gap != 0:
            if (timepeaks[i + 1] - timepeaks[i]) > gap:
                peakstr += '#' * ((timepeaks[i + 1] - timepeaks[i]) /gap)

    if part < npart+1:
        rfile = open(datainfo.dpath+ '/'+ datainfo.name + '/Results/stringseq-%s%d-%s-%s-%d.txt'%(nfile, part, ename, sensor, ncl), 'w')
        for i in range(0, len(peakstr), 250):
            wstr = ''
            for j in range(250):
                if i+j < len(peakstr):
                    wstr += peakstr[i+j]
            rfile.write(wstr + '\n')
        rfile.close()

    return peakstr


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


def partition_experiment(expnames):
    """
    partitions an experiment in sections according the the three first characters of the expnames
    :param expnames:
    :return:
    """

    lpart = []
    lpartact = [(expnames[0], 0)]
    for iname in range(1, len(expnames)):
        if expnames[iname-1][0:3] == expnames[iname][0:3]:
            lpartact.append((expnames[iname], iname))
        else:
            lpart.append(lpartact)
            lpartact = [(expnames[iname], iname)]

    lpart.append(lpartact)

    return(lpart)


def frequent_strings_intersection(lfrstrings):
    """
    intersection of frequent pairs of peaks

    :param lfrstrings:
    :return:
    """
    base = lfrstrings[0]
    result = []

    for elem in base:
        memb = True
        for cnj in lfrstrings[1:]:
            if elem not in cnj:
                memb = False
                break
        if memb:
            result.append(elem)

    return [(r, 0.0, 0.0) for r in result]

def compute_intersection_graphs(datainfo, exppartition, lfrstrings, ncl, sensor, lmatch=0, mapping=None, galt=False, partition=None):
    """
    Computes the intersection graph among files from the same type of experiment
    """

    ext = '' if lmatch == 0 else '-match'
    for part in exppartition:
        lfreq = []
        for _, i in part:
            lfreq.append([st for st, _, _ in lfrstrings[i]])

        inter = frequent_strings_intersection(lfreq)
        nfile = datainfo.name + '-inter'
        ename = part[0][0]
        dfile = ''
        drawgraph_alternative(ncl, inter, nfile, sensor, dfile, ename,
                                  nfile + '-' + ename + '-' + sensor + ext,
                                  partition=partition, lmatch=lmatch, mapping=mapping)

        for i in range(len(lfreq)-1):
            inter = frequent_strings_intersection([lfreq[i], lfreq[i+1]])
            nfile = datainfo.name + '-inter-seq'
            ename = part[i][0] + '-'  + part[i+1][0]
            drawgraph_alternative(ncl, inter, nfile, sensor, dfile, ename,
                                      nfile + '-' + part[i][0] + '-'  + part[i+1][0] + '-' + sensor + ext,
                                      partition=partition, lmatch=lmatch, mapping=mapping)

        for i in range(2, len(lfreq)+1):
            inter = frequent_strings_intersection(lfreq[:i])
            nfile = datainfo.name + '-inter-sub'

            ename = '-'.join([v[0] for v in part[:i]])
            drawgraph_alternative(ncl, inter, nfile, sensor, dfile, ename,
                                      nfile + '-' + ename + '-' + sensor + ext,
                                      partition=partition, lmatch=lmatch, mapping=mapping)


# ----------------------------------------


def generate_partition(nvals, npart, colors):
    partition = []
    div = nvals/npart
    if nvals % npart != 0:
        div += 1
    for i in range(npart):
        partition.append([range(i*div, min((i*div)+div, nvals)), colors[i]])
    return partition

voc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*+-$%&/<>[]{}()!?#'



def plot_pairs_dist(datainfo, sensor, lacounts):

    fig = plt.figure()
    #plt.plot(np.cumsum(sorted(acounts)))
    #plt.plot(sorted(acounts))
    lcol = np.unique(list(datainfo.colors))

    for j, col in enumerate(lcol):
        sp1 = fig.add_subplot(len(lcol), 1, j + 1)
        sp1.axis([0, len(lacounts[0]), 0, 0.06])

        for i, acounts in enumerate(lacounts):
            if datainfo.colors[i] == col:
                t = arange(0.0, len(acounts), 1)
                plt.plot(t, acounts, color=datainfo.colors[i])

    plt.suptitle(datainfo.name + '-' + sensor)
    plt.show()


# --------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--graph', help="circular graph of the sequences", action='store_true', default=False)
    parser.add_argument('--gpropor', help="Arrows of circular graph are proportional to the probabilities", action='store_true', default=False)
    parser.add_argument('--freqstr', help="List with the frequent strings and their probabilities", action='store_true', default=False)
    parser.add_argument('--contingency', help="Contingency matrix of the frequent pairs", action='store_true', default=False)
    parser.add_argument('--sequence', help="linear graph of the sequences", action='store_true', default=False)
    parser.add_argument('--string', help="generate a string representation of the sequences", action='store_true', default=False)
    parser.add_argument('--galternative', help="Alternative Coloring for the Graph", action='store_true', default=True)
    parser.add_argument('--matching', help="Perform matching of the peaks", action='store_true', default=False)
    parser.add_argument('--rescale', help="Rescale the peaks for matching", action='store_true', default=False)
    parser.add_argument('--diffs', help="Computes the differences among circular graphs", action='store_true', default=False)
    parser.add_argument('--globalclust', help="Use a global computed clustering", action='store_true', default=False)

    args = parser.parse_args()
    lexperiments = args.exp

    if not args.batch:
        args.graph = True
        args.freqstr = True
        args.contingency = True
        args.sequence = True
        args.matching = False
        args.rescale = False
        args.string = True
        args.galternative = True
        args.diffs = False
        args.globalclust = False
        args.gpropor = False
        # 'e120503''e110616''e150707''e151126''e120511','e151126''e120511', 'e120503', 'e110906o', 'e160204''e150514'
        lexperiments = ['e160317']

    colors = ['red', 'blue', 'green']
    npart = 3
    gap = 2000
    sup = None
    rand = False

    # Matching parameters
    isig = 2
    fsig = 10

    peakdata = {}
    for expname in lexperiments:

        datainfo = experiments[expname]
        exppartition = partition_experiment(datainfo.expnames)

        f = datainfo.open_experiment_data(mode='r')
        if args.matching:
            lsensors = datainfo.sensors[isig:fsig]
            lclusters = datainfo.clusters[isig:fsig]
            smatching = compute_signals_matching(datainfo, lsensors, rescale=args.rescale, globalc=args.globalclust)
            print len(smatching)
        else:
            lsensors = datainfo.sensors
            lclusters = datainfo.clusters
            smatching = []

        for nclusters, sensor in zip(lclusters, lsensors):
            print(sensor)
            if args.matching:
                mapping = compute_matching_mapping(nclusters, sensor, smatching)
            else:
                mapping = None

            lfrstrings = []
            lacounts = []
            for dfile, ename in zip(datainfo.datafiles, datainfo.expnames):
                print(dfile, ename)

                d = datainfo.get_peaks_time(f, dfile, sensor)
                if d is not None:
                    clpeaks = datainfo.compute_peaks_labels(f, dfile, sensor, nclusters, globalc=args.globalclust)
                    timepeaks = d[()]

                    peakstr, peakfreq, lstrings = peaks_sequence_frequent_strings(timepeaks, gap=gap, rand=rand, sup=sup)

                    if args.graph or args.freqstr or args.contingency:
                        if len(smatching) != 0:
                            partition = generate_partition(len(smatching), npart, colors)
                        else:
                            partition = generate_partition(nclusters, npart, colors)

                        fstrings = save_frequent_sequences(dfile, peakstr, peakfreq, lstrings, sensor, dfile, ename, nclusters,
                                                           lmatch=len(smatching), mapping=mapping, rand=rand, galt=args.galternative,
                                                           partition=partition, sup=sup, save=(args.freqstr, args.contingency, args.graph),
                                                           proportional=args.gpropor, globalc=args.globalclust)
                        lfrstrings.append(fstrings)
                        # generate_sequences(dfile, ename, timepeaks, clpeaks, sensor, ncl,
                        #                    lmatch=len(smatching), mapping=mapping,
                        #                    gap=gap, sup=None, rand=False, galt=args.galternative, partition=partition)

                    if args.sequence:
                        lseq = freq_seq_positions(datainfo.name, clpeaks, timepeaks, sensor, ename, nclusters,
                                                  gap=gap, sup=sup)
                        plot_sequences(dfile, lseq, nclusters, sensor, lmatch=len(smatching), mapping=mapping)

                    if args.string:
                        sqstr = sequence_to_string(dfile, clpeaks, timepeaks, sensor, ename, gap=gap, npart=1, ncl=nclusters)
                        if len(smatching) != 0:
                            lacounts.append(compute_pairs_distribution(sqstr, len(smatching)))
                        else:
                            lacounts.append(compute_pairs_distribution(sqstr, nclusters))

            #plot_pairs_dist(datainfo, sensor, lacounts)
            if args.diffs:
                compute_intersection_graphs(datainfo, exppartition, lfrstrings, nclusters, sensor, lmatch=len(smatching), mapping=mapping,
                                            galt=args.galternative, partition=partition)
        datainfo.close_experiment_data(f)