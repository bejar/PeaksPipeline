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

__author__ = 'bejar'

import h5py

from Config.experiments import experiments
from pylab import *
import seaborn as sns
from Secuencias.rstr_max import *
from util.misc import compute_frequency_remap
from sklearn.metrics import pairwise_distances_argmin_min
import random
import string
import os
import argparse
from pyx import *
import operator

from util.misc import choose_color
from Matching.Match import compute_matching_mapping, compute_signals_matching

def randomize_string(s):
    l = list(s)
    random.shuffle(l)
    result = ''.join(l)
    return result

def drawgraph_alternative(nnodes, edges, nfile, sensor, dfile, ename, legend, partition, lmatch=0, mapping=None):
    rfile = open(datainfo.dpath + '/' + datainfo.name + '/Results/maxseqAlt-' + nfile + '-' + dfile + '-' + sensor + '-' + ename + '.dot', 'w')

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
                    ', image="' + datainfo.name + sensor + '.cl' + str(i+1) + '.png' + '"' +
                    ', pos="' + str(posx) + ',' + str(posy) + '!", shape = "square"];\n')

    for e, nb, pe in edges:
        if len(e) == 2:
            rfile.write(str(e[0]) + '->' + str(e[1]))
            for lelem, color in partition:
                if lmatch != 0:
                    if mapping[e[0]] in lelem:
                        rfile.write('[color="'+color+'"]')
                else:
                    if e[0] in lelem:
                        rfile.write('[color="'+color+'"]')

            rfile.write('\n')


    rfile.write('}\n')

    rfile.close()
    os.system('dot -Tpdf '+datainfo.dpath + '/'+ datainfo.name+ '/Results/maxseqAlt-' + nfile + '-' + dfile + '-' + sensor + '-' + ename+ '.dot ' + '-o '
              + datainfo.dpath + '/'+ datainfo.name+ '/Results/maxseqAlt-' + nfile + '-' + dfile + '-' + sensor + '-' + ename + '.pdf')
    os.system(' rm -fr ' + datainfo.dpath + '/'+ datainfo.name+ '/Results/maxseqAlt-' + nfile + '-' + dfile + '-' + sensor + '-' + ename + '.dot')


def drawgraph(nnodes, edges, nfile, sensor, dfile, legend, lmatch=0, mapping=None):
    rfile = open(datainfo.dpath + '/' + datainfo.name + '/Results/maxseq-' + nfile + '-' + dfile + '-' + sensor + '.dot', 'w')

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

    d.writePDFfile(datainfo.dpath + '/' + datainfo.name + "/Results/peaksseq-%s-%s-%s-%s" % (datainfo.name, dfile, sensor, ename))


def max_seq_exp(nfile, clpeaks, timepeaks, sensor, dfile, ename, nclust,
                lmatch=0, mapping=None,
                gap=0, sup=None, rand=False, galt=False, partition=None):
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

    # Support computed heuristically
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
    if lmatch !=0:
        mfreqmatch = np.zeros((lmatch, lmatch))
    for seq, s in lstrings:
        wstr = ''
        prob = 1.0
        if not '#' in seq:
            if len(seq) == 2:
                mfreq[voc.find(seq[0]),  voc.find(seq[1])] = int(s)
                if lmatch !=0:
                    mfreqmatch[mapping[voc.find(seq[0])],  mapping[voc.find(seq[1])]] = int(s)


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

    # Contingency table of the number of times a frequent sequence of length 2 has appeared
    fig = plt.figure()

    if lmatch !=0:
        sns.heatmap(mfreqmatch, annot=True, fmt='.0f', cbar=False, xticklabels=range(1,lmatch+1), yticklabels=range(1,lmatch+1), square=True)
    else:
        sns.heatmap(mfreq, annot=True, fmt='.0f', cbar=False, xticklabels=range(1,nclust+1), yticklabels=range(1,nclust+1), square=True)
    plt.title(nfile + '-' + ename + '-' + sensor + ' sup(%d)' % sup)
    plt.savefig(datainfo.dpath + '/' + datainfo.name + '/Results/maxseq-histo' + datainfo.name + '-' + dfile + '-'
                + sensor  + '-freq.pdf', orientation='landscape', format='pdf')
    plt.close()
    # ----------------


    # Circular graph of the frequent sequences of length 2
    nsig = len(peakfreq)
    if '#' in peakfreq:
        nsig -= 1

    if galt:
        drawgraph_alternative(nclust, lstringsg, nfile, sensor, dfile, ename,
                              nfile + '-' + ename + '-' + sensor + ' sup(%d)' % sup,
                              partition=partition, lmatch=lmatch, mapping=mapping)
        # drawgraph_alternative(nclust, lstringsg, nfile, sensor, dfile, ename,
        #                       sensor+'-'+ename,
        #                       partition=partition, lmatch=lmatch, mapping=mapping)
    else:
        drawgraph(nclust, lstringsg, nfile, sensor, dfile,
                  nfile + '-' + ename + '-' + sensor + ' sup(%d)' % sup,
                  lmatch=lmatch, mapping=mapping)


def sequence_to_string(nfile, clpeaks, timepeaks, sensor, dfile, ename, gap=0, npart=1):
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

    mtime = timepeaks[-1]/npart
    peakstr = ''
    part = 1
    for i in range(timepeaks.shape[0]):
        if timepeaks[i] > part*mtime:
            rfile = open(datainfo.dpath + '/' + datainfo.name + '/Results/stringseq-%s%d-%s-%s.txt'%(nfile, part, ename, sensor), 'w')
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
        rfile = open(datainfo.dpath+ '/'+ datainfo.name + '/Results/stringseq-%s%d-%s-%s.txt'%(nfile, part, ename, sensor), 'w')
        for i in range(0, len(peakstr), 250):
            wstr = ''
            for j in range(250):
                if i+j < len(peakstr):
                    wstr += peakstr[i+j]
            rfile.write(wstr + '\n')
        rfile.close()


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


def generate_sequences(dfile, ename, timepeaks, clpeaks, sensor, ncl, gap,
                       lmatch=0, mapping=None, sup=None, rand=False, galt=False, partition=None):
    """
    Generates the frequent subsequences from the times of the peaks considering
    gap the minimum time between consecutive peaks that indicates a pause (time in the sampling frequency)

    :param dfile:
    :param timepeaks:
    :param clpeaks:
    :param sensor:
    :return:
    """
    max_seq_exp(datainfo.name, clpeaks, timepeaks, sensor, dfile, ename, ncl,
                lmatch=lmatch, mapping=mapping,
                gap=gap, sup=sup, rand=rand, galt=galt, partition=partition)


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


def generate_partition(nvals, npart, colors):
    partition = []
    div = nvals/npart
    if nvals % npart != 0:
        div += 1
    for i in range(npart):
        partition.append([range(i*div, min((i*div)+div, nvals)), colors[i]])
    return partition

voc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*+-$%&/<>[]{}()!?#'
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--graph', help="circular graph of the sequences", action='store_true', default=True)
    parser.add_argument('--sequence', help="linear graph of the sequences", action='store_true', default=True)
    parser.add_argument('--string', help="generate a string representation of the sequences", action='store_true', default=True)
    parser.add_argument('--alternative', help="Alternative Coloring for the Graph", action='store_true', default=True)
    parser.add_argument('--matching', help="Perform matching of the peaks", action='store_true', default=False)
    parser.add_argument('--rescale', help="Rescale the peaks for matching", action='store_true', default=False)

    args = parser.parse_args()
    lexperiments = args.exp

    if not args.batch:
        args.graph = True
        args.sequence = True
        args.matching = False
        args.rescale = False
        args.string = True
        # 'e150514''e120503''e110616''e150707''e151126''e120511', 'e150707', 'e151126''e120511', 'e120503'
        lexperiments = ['e160204']

    galt = args.alternative

    colors = ['red', 'blue', 'green']
    npart = 3

    # Matching parameters
    isig = 2
    fsig = 10

    peakdata = {}
    for expname in lexperiments:

        datainfo = experiments[expname]
        f = h5py.File(datainfo.dpath + '/' + datainfo.name + '/' + datainfo.name + '.hdf5', 'r')
        if args.matching:
            lsensors = datainfo.sensors[isig:fsig]
            lclusters = datainfo.clusters[isig:fsig]
            smatching = compute_signals_matching(expname, lsensors, rescale=args.rescale)
            print len(smatching)
        else:
            lsensors = datainfo.sensors
            lclusters = datainfo.clusters
            smatching = []

        for dfile, ename in zip(datainfo.datafiles, datainfo.expnames):
            print(dfile)

            for ncl, sensor in zip(lclusters, lsensors):
                if dfile + '/' + sensor + '/' + 'Time' in f:
                    if args.matching:
                        mapping = compute_matching_mapping(ncl, sensor, smatching)
                    else:
                        mapping=None

                    clpeaks = compute_data_labels(datainfo.datafiles[0], dfile, sensor)
                    d = f[dfile + '/' + sensor + '/' + 'Time']
                    timepeaks = data = d[()]
                    if args.graph:
                        if len(smatching)!= 0:
                            partition = generate_partition(len(smatching), npart, colors)
                        else:
                            partition = generate_partition(ncl, npart, colors)

                        generate_sequences(dfile, ename, timepeaks, clpeaks, sensor, ncl,
                                           lmatch=len(smatching), mapping=mapping,
                                           gap=2000, sup=None, rand=False, galt=galt, partition=partition)

                    if args.sequence:
                        lseq = freq_seq_positions(datainfo.name, clpeaks, timepeaks, sensor, ename, ncl,
                                                   gap=2000, sup=None)
                        plot_sequences(dfile, lseq, ncl, sensor, lmatch=len(smatching), mapping=mapping)

                    if args.string:
                        sequence_to_string(dfile, clpeaks, timepeaks, sensor, dfile, ename, gap=2000, npart=1)
