"""
.. module:: PeaksSequenceSegmentation

PeaksSequenceSegmentation
*************

:Description: PeaksSequenceSegmentation

    Signal segmentation following a Page-Hinkley approach

:Authors: bejar
    

:Version: 

:Created on: 01/06/2016 10:20 

"""

from Config.experiments import experiments
from pylab import *
import seaborn as sns
from sklearn.metrics import pairwise_distances_argmin_min
import os
import argparse
import operator
from util.distances import jensen_shannon_divergence, hellinger_distance
__author__ = 'bejar'



def add_gaps(peaks, times, gap=0):
    """
    Generates a sequence with the peaks and the gaps

    :param peaks:
    :param times:
    :return:
    """

    sequence = []
    for i in range(len(peaks)-1):
        sequence.append(np.array([peaks[i], times[i]]))
        if (times[i+1] - times[i] > gap):
            for j in range(times[i]+gap, times[i+1], gap):
                sequence.append(np.array([-1, j]))
    return sequence


def estimate_frequency_model(sequence, nclusters, laplace=0):
    """
    Estimation of a multinomial model as frequencies of pairs
    it uses the estsize

    Mind of the gap :-)
    :return:
    """

    counts_model = np.zeros(nclusters+1)

    for i in range(len(sequence)):
        counts_model[sequence[i][0]+1] += 1

    return counts_model+laplace


def estimate_frequency_pairs_model(sequence, nclusters, laplace=0):
    """
    Estimation of a multinomial model as frequencies of pairs
    it uses the estsize

    Mind of the gap :-)
    :return:
    """

    counts_model = np.zeros((nclusters+1, nclusters+1))

    for i in range(len(sequence)-1):
        counts_model[sequence[i][0]+1][sequence[i+1][0]+1] += 1

    return counts_model+laplace

def reestimate_multinomial_model(counts_model, seq_out, seq_in):
    """
    Reestimates the frequency matrix deleting and adding pairs

    :param seq_out:
    :param seq_in:
    :return:
    """
    for i in range(len(seq_out)-1):
        counts_model[seq_out[i][0]+1][seq_out[i+1][0]+1] -= 1

    for i in range(len(seq_in)-1):
        counts_model[seq_in[i][0]+1][seq_in[i+1][0]+1] += 1

    return counts_model

# --------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--pairs', help="Estimate sequence pairs frequecy model", action='store_true', default=True)
    parser.add_argument('--globalclust', help="Use a global computed clustering", action='store_true', default=False)


    args = parser.parse_args()
    lexperiments = args.exp

    if not args.batch:
        # 'e150514''e150707''e160204''e151126''e160317','e110906o','e120511', 'e140225'
        lexperiments = ['e160204']
        args.globalclust = False
        args.pairs = False
    gap = 2000  # Assuming 10KHz frequency
    laplace = 0.001  # Laplace smoothing of the probability estimation
    # Tolerance in the difference between two models
    tolerance = 0.9

    for expname in lexperiments:

        print(expname)
        datainfo = experiments[expname]
        lsensors = datainfo.sensors
        lclusters = datainfo.clusters
        f = datainfo.open_experiment_data(mode='r')

        fig = plt.figure()
        fig.set_figwidth(20)
        fig.set_figheight(30)
        fig.suptitle(expname + '-T'+str(tolerance), fontsize=48)

        nfig = 0
        nfil = len(lsensors)/2
        if nfil % 2 == 1:
            nfil +=1
        ncol = 2
        #plt.subplots_adjust(top=0.95)

        for nclusters, sensor in zip(lclusters, lsensors):
            nfig += 1
            print(sensor)

            sequences = []
            seqend = []
            # Append all the experiments sequences
            for dfile, ename in zip(datainfo.datafiles, datainfo.expnames):
                #print(dfile, ename)
                d = datainfo.get_peaks_time(f, dfile, sensor)
                if d is not None:
                    clpeaks = datainfo.compute_peaks_labels(f, dfile, sensor, nclusters, globalc=args.globalclust)
                    timepeaks = d[()]
                    #sequences.append([ename, add_gaps(clpeaks, timepeaks, gap=gap)])
                    sequences.extend(add_gaps(clpeaks, timepeaks, gap=gap))
                    seqend.append(len(sequences))

            # Length of the initial sequence for model estimation
            estsize = nclusters * nclusters * 10
            # Step size to advance in the sequence
            stepsize = nclusters * nclusters * 1


            # Base model
            lldiff = []
            llmark = []
            llend = []
            tkpos = []
            tk = 0

            if args.pairs:
                counts_model = estimate_frequency_pairs_model(sequences[0:estsize], nclusters, laplace=laplace)
            else:
                counts_model = estimate_frequency_model(sequences[0:estsize], nclusters, laplace=laplace)


            for i in range(0, len(sequences)-stepsize, stepsize):
                tk += 1
                if args.pairs:
                    counts_model_ahead = estimate_frequency_pairs_model(sequences[i:i + estsize], nclusters, laplace=laplace)
                else:
                    counts_model_ahead = estimate_frequency_model(sequences[i:i + estsize], nclusters, laplace=laplace)

                #diff = jensen_shannon_divergence(counts_model/sum(counts_model), counts_model_ahead/sum(counts_model_ahead))
                diff = hellinger_distance(counts_model/sum(counts_model), counts_model_ahead/sum(counts_model_ahead))

                lldiff.append(diff)
                if diff>tolerance:
                    counts_model = counts_model_ahead.copy()
                    llmark.append(tolerance*1.2)
                else:
                    llmark.append(0)
                if i > seqend[0]:
                    seqend.pop(0)
                    tkpos.append(tk)
            tkpos.insert(0, 0)

            sp1 = fig.add_subplot(nfil, ncol, nfig)
            sp1.axis([0, len(lldiff), 0, tolerance*1.2])
            plt.title(sensor)
            #fig.suptitle(sensor, fontsize=48)
            sp1.plot(range(len(lldiff)), lldiff, 'b')
            sp1.plot(range(len(llmark)), llmark, 'r')
            plt.xticks(tkpos, datainfo.expnames, rotation='vertical')

        plt.tight_layout()

        if args.pairs:
            pairsp = '-Ppairs'
        else:
            pairsp = '-Psingle'

        fig.savefig(datainfo.dpath + '/' + datainfo.name + '/Results/' + datainfo.name +
                   '-segment-S' + str(tolerance) + pairsp + '.pdf', orientation='landscape', format='pdf')

        #plt.show()
        #plt.close()


