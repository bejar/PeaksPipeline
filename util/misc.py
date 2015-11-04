"""
.. module:: misc

misc
*************

:Description: misc

    

:Authors: bejar
    

:Version: 

:Created on: 17/11/2014 13:37 

"""

__author__ = 'bejar'

from operator import itemgetter

import numpy as np


def peaks_sequence(clpeaks, timepeaks, nexp, peakini, peakend, gap):
    """
    Returns a sequence from the peaks as a list with the class of the peak and the moment of the peak.
    It also returns the count of peaks

    :param clpeaks:
    :param timepeaks:
    :param nexp:
    :param peakini:
    :param peakend:
    :param gap:
    :return:
    """
    peakfreq = {}
    peakstr = []

    for i in range(peakini, peakend):
        # Insert in the list the class of the peak and its time
        peakstr.append((clpeaks[i][0], timepeaks[nexp][i - peakini][0]))

        # Counts the peaks
        if clpeaks[i][0] in peakfreq:
            peakfreq[clpeaks[i][0]] += 1
        else:
            peakfreq[clpeaks[i][0]] = 1
    return peakstr, peakfreq


def find_time_end(seq, ini, step):
    """
    Finds the index of the peak that is at step time from the ini peak
    Seq is a sequence of pairs (peak, time)

    :param exp:
    :param ini:
    :return:
    """
    i = ini + 1
    while i < len(seq) and seq[i][1] < seq[ini][1] + step:
        i += 1
    return i


def probability_matrix_seq(peakseq, init, end, nsym, gap, laplace=0.0):
    """
    Computes the probability matrix of the transitions from a sequence of points
    Each element of the sequence is the class and the moment of time of the peak

    This function ignores the times of the peaks

    The Laplace correction assumes a minimum number of transitions per peak

    :param peakseq:
    :param init:
    :param end:
    :return:
    """

    pm = np.zeros((nsym, nsym)) + laplace
    for i in range(init, end - 1):
        if peakseq[i + 1][1] - peakseq[i][1] < gap:
            pm[peakseq[i][0] - 1, peakseq[i + 1][0] - 1] += 1.0
    return pm / pm.sum()


def probability_matrix_multi(peakseq, init, end, nsym, gap, laplace=0.0):
    """
    Computes the probability matrix of the transitions from a sequence of points
    between two points of time.

    Each element of the sequence is the class and the moment of time of the peak

    This function assumes that one peak can affect up to to a horizon  of
    peaks until the gap is reached

    The Laplace correction assumes a minimum number of transitions per peak

    :param peakseq:
    :param init:
    :param end:
    :return:
    """

    pm = np.zeros((nsym, nsym)) + laplace
    for i in range(init, end - 1):
        j = i + 1
        lprob = []
        sm = 0.0
        while j < end - 1 and peakseq[j][1] - peakseq[i][1] < gap:
            sm += (gap - (peakseq[j][1] - peakseq[i][1]))
            lprob.append((peakseq[i][0], gap - (peakseq[j][1] - peakseq[i][1])))
            j += 1
        for pk, pr in lprob:
            pm[peakseq[i][0] - 1, pk - 1] += (pr / sm)

    return pm / pm.sum()


def normalize_matrix(matrix):
    """
    Normalizes all the rows of the matrix to sum one
    :param matrix:
    :return:
    """
    for i in range(matrix.shape[0]):
        matrix[i] /= matrix[i].sum()
    return matrix


def compute_frequency_remap(timepeaks, clpeaks):
    """
    Computes the remapping of the indices of the peaks according to their frequency on the
    first experiment

    :param timepeaks:
    :param clpeaks:
    :return:
    """
    clstfreq = {}
    for i in range(0, timepeaks[0].shape[0]):
        if clpeaks[i][0] in clstfreq:
            clstfreq[clpeaks[i][0]] += 1
        else:
            clstfreq[clpeaks[i][0]] = 1

    lclstfreq = [(k, clstfreq[k]) for k in clstfreq]
    lclstfreq = sorted(lclstfreq, key=itemgetter(1), reverse=True)
    remap = [i for i, _ in lclstfreq]
    return remap

