"""
.. module:: plots

plots
*************

:Description: plots

    

:Authors: bejar
    

:Version: 

:Created on: 17/11/2014 13:40 

"""


import matplotlib.pyplot as plt
import matplotlib
import numpy as np


__author__ = 'bejar'


def plotHungarianSignals(coinc, centers1, centers2, vmax, vmin, name, title, path):
    """
    Plots two sets of signals according to the Hungarian Algorithm assignment
    :param coinc:
    :param centers1:
    :param centers2:
    :param name:
    :param title:
    :param path:
    :return:
    """

    matplotlib.rcParams.update({'font.size': 26})
    fig = plt.figure()
    fig.set_figwidth(30)
    fig.set_figheight(40)
    i = 1
    n = len(coinc)
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    for i1, i2 in coinc:
        if i1 < centers1.shape[0]:
            s1 = centers1[i1]
            plotSignalValues(fig, s1, n, 2, i, 'cluster' + str(i1), vmax, vmin)
        else:
            plotDummy(fig, centers1.shape[0], n, 2, i, '')

        if i2 < centers2.shape[0]:
            s2 = centers2[i2]

            plotSignalValues(fig, s2, n, 2, i + 1, 'cluster' + str(i2), vmax, vmin)
        else:
            plotDummy(fig, centers2.shape[0], n, 2, i + 1, '')
        i += 2

    fig.suptitle(title, fontsize=48)
    fig.savefig(path + '/' + name + '.pdf', orientation='landscape', format='pdf')
    plt.close()


def plotSignals(signals, n, m, vmax, vmin, name, title, path, cstd=None, orientation='landscape'):
    """
     Plots list of signals (signal,name)

    :param signals:
    :param n:
    :param m:
    :param vmax:
    :param vmin:
    :param name:
    :param title:
    :param path:
    :param cstd:
    :return:
    """
    matplotlib.rcParams.update({'font.size': 26})
    fig = plt.figure()
    if orientation == 'landscape':
        fig.set_figwidth(30)
        fig.set_figheight(40)
    else:
        fig.set_figwidth(40)
        fig.set_figheight(30)

    i = 1
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    for s, snm in signals:
        if min(s) != max(s):
            if cstd is not None:
                pstd = cstd[i - 1]
            else:
                pstd = None
            plotSignalValues(fig, s, n, m, i, snm, vmax, vmin, cstd=pstd)
        else:
            plotDummy(fig, len(s), n, m, i, snm)
        i += 1

    fig.suptitle(title, fontsize=48)
    fig.savefig(path + '/' + name + '.pdf', orientation=orientation, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()


# plt.show()

def show_signal(signal, line=None, title=''):
    """
    Plots a signal
    :param title:
    :param signal:
    :return:
    """
    matplotlib.rcParams.update({'font.size': 26})
    fig = plt.figure()
    fig.set_figwidth(30)
    fig.set_figheight(40)
    fig.suptitle(title, fontsize=48)
    minaxis = min(signal)
    maxaxis = max(signal)
    num = len(signal)
    sp1 = fig.add_subplot(111)
    sp1.axis([0, num, minaxis, maxaxis])
    t = np.arange(0.0, num, 1)
    sp1.plot(t, signal)
    if line is not None:
        plt.axhline(linewidth=4, color='r', y=line)
    plt.show()
    plt.close()


def show_vsignals(signal, title=''):
    """
    Plots a list of signals
    :param title:
    :param signal:
    :return:
    """
    matplotlib.rcParams.update({'font.size': 26})
    fig = plt.figure()
    fig.set_figwidth(30)
    fig.set_figheight(40)
    fig.suptitle(title, fontsize=48)
    minaxis = np.min(signal)
    maxaxis = np.max(signal)
    num = signal.shape[1]
    sp1 = fig.add_subplot(111)
    sp1.axis([0, num, minaxis, maxaxis])
    t = np.arange(0.0, num, 1)
    for i in range(signal.shape[0]):
        sp1.plot(t, signal[i, :])
    plt.show()


def show_two_signals(signal1, signal2):
    """
    Shows two signals in the same picture
    :param signal1:
    :param signal2:
    :return:
    """
    fig = plt.figure()
    fig.set_figwidth(30)
    fig.set_figheight(40)
    minaxis = np.min([np.min(signal1), np.min(signal2)])
    maxaxis = np.max([np.max(signal1), np.max(signal2)])
    num = len(signal1)
    sp1 = fig.add_subplot(111)
    sp1.axis([0, num, minaxis, maxaxis])
    t = np.arange(0.0, num, 1)
    sp1.plot(t, signal1, 'r')
    sp1.plot(t, signal2, 'b')
    plt.show()

def show_list_signals(signals, legend=[]):
    """
    Shows a list of signals in the same picture
    :param signal1:
    :param signal2:
    :return:
    """
    cols = ['r', 'g', 'b', 'k', 'y', 'c']
    fig = plt.figure()
    fig.set_figwidth(30)
    fig.set_figheight(40)

    minaxis = np.min([np.min(s) for s in signals])
    maxaxis = np.max([np.max(s) for s in signals])
    num = len(signals[0])
    sp1 = fig.add_subplot(111)
    sp1.axis([0, num, minaxis, maxaxis])
    t = np.arange(0.0, num, 1)
    for i, s in enumerate(signals):
        sp1.plot(t, s, cols[i])
    plt.legend(legend)
    plt.show()


def plotSignalValues(fig, signal1, n, m, p, name, vmax, vmin, cstd=None):
    """
    Plot a set of signals
    """
    minaxis = vmin  # min(signal1)
    maxaxis = vmax  # max(signal1)
    num = len(signal1)
    sp1 = fig.add_subplot(n, m, p)
    plt.title(name)
    sp1.axis([0, num, minaxis, maxaxis])
    t = np.arange(0.0, num, 1)
    if cstd == 'mv':
        plt.axhline(linewidth=4, color='r', y=np.mean(signal1))
        pstd = np.std(signal1)
        plt.axhline(linewidth=4, color='b', y=np.mean(signal1) + pstd)
        plt.axhline(linewidth=4, color='b', y=np.mean(signal1) - pstd)
        sp1.plot(t, signal1 + cstd)
        sp1.plot(t, signal1 - cstd)
    elif type(cstd) == float:
        plt.axhline(linewidth=1, color='r', y=cstd)
    sp1.plot(t, signal1)


def plotListSignals(signals, orient='h', ncols=None):
    """
    Plot a set of signals
    """
    fig = plt.figure()
    minaxis = np.min([np.min(s) for s in signals])
    maxaxis = np.max([np.max(s) for s in signals])
    num = len(signals)
    if ncols is not None:
        if num % ncols == 0:
            nrows = num / ncols
        else:
            nrows = (num / ncols) + 1
    for i in range(num):
        if ncols is not None:
            sp1 = fig.add_subplot(nrows, ncols, i + 1)
        elif orient == 'h':
            sp1 = fig.add_subplot(1, num, i + 1)
        elif orient == 'v':
            sp1 = fig.add_subplot(num, 1, i + 1)
        sp1.axis([0, signals[0].shape[0], minaxis, maxaxis])
        t = np.arange(0.0, signals[0].shape[0], 1)
        sp1.plot(t, signals[i])
    plt.show()
    plt.close()


def plotParallelSignals(signals1, signals2):
    """
    Plot a set of signals
    """
    fig = plt.figure()
    minaxis1 = np.min([np.min(s) for s in signals1])
    maxaxis1 = np.max([np.max(s) for s in signals1])
    minaxis2 = np.min([np.min(s) for s in signals2])
    maxaxis2 = np.max([np.max(s) for s in signals2])


    minaxis = min(minaxis1, minaxis2)
    maxaxis = max(maxaxis1, maxaxis2)

    num = len(signals1)

    for i in range(num):
        sp1 = fig.add_subplot(num, 2, (2*i) + 1)
        sp1.axis([0, signals1[0].shape[0], minaxis, maxaxis])
        t = np.arange(0.0, signals1[0].shape[0], 1)
        sp1.plot(t, signals1[i])

    for i in range(num):
        sp1 = fig.add_subplot(num, 2, (2*i) + 2)
        sp1.axis([0, signals1[0].shape[0], minaxis, maxaxis])
        t = np.arange(0.0, signals1[0].shape[0], 1)
        sp1.plot(t, signals2[i])

    plt.show()
    plt.close()

def plotDummy(fig, num, n, m, p, name):
    minaxis = -1
    maxaxis = 1
    sp1 = fig.add_subplot(n, m, p)
    plt.title(name)
    sp1.axis([0, num, minaxis, maxaxis])
    t = np.arange(0.0, num, 1)
    sp1.plot(t, t)


#    plt.show()


def plotMatrices(matrices, n, m, name, title, path, ticks=[], lticks=[]):
    """
    Plots a list of matrices

    :param matrices: pairs matrix, title of the plot of the matrix
    :param n:
    :param m:
    :param name:
    :param title:
    :param path:
    :return:
    """
    matplotlib.rcParams.update({'font.size': 32})
    fig = plt.figure()
    fig.set_figwidth(50)
    fig.set_figheight(60)
    i = 1
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    for s, snm in matrices:
        if s is not None:
            plotMatrixValues(fig, s, n, m, i, snm, ticks=ticks, lticks=lticks)
        else:
            plotMatrixDummy(fig, len(s), n, m, i, snm)
        i += 1

    fig.suptitle(title, fontsize=60)
    fig.savefig(path + '/' + name + '.pdf', orientation='landscape', format='pdf')
    plt.close()


#    plt.show()

def plotMatrixValues(fig, matrix, n, m, p, name, ticks=[], lticks=[]):
    """
    Plot a set of signals
    """
    sp1 = fig.add_subplot(n, m, p)
    plt.title(name, fontsize=48)
    sp1.imshow(matrix, cmap=plt.cm.gray, interpolation='none')
    plt.xticks(ticks, lticks, fontsize=40)
    plt.yticks(ticks, lticks, fontsize=40)


#    plt.show()

def plotMatrixDummy(fig, num, n, m, p, name):
    minaxis = -1
    maxaxis = 1
    sp1 = fig.add_subplot(n, m, p)
    plt.title(name)
    sp1.axis([0, num, minaxis, maxaxis])
    t = np.arange(0.0, num, 1)
    sp1.plot(t, t)


#   plt.show()


def plotMatrix(matrix, name, title, ticks, lticks, path):
    """
    """
    matplotlib.rcParams.update({'font.size': 40})
    fig = plt.figure()
    fig.set_figwidth(50)
    fig.set_figheight(50)
    sp1 = fig.add_subplot(1, 1, 1)
    plt.title(title, fontsize=48)
    img = sp1.imshow(matrix, cmap=plt.cm.seismic, interpolation='none')
    plt.xticks(ticks, lticks, fontsize=40)
    plt.yticks(ticks, lticks, fontsize=40)
    plt.subplots_adjust(bottom=0.15)
    fig.colorbar(img, orientation='horizontal')
    fig.savefig(path + '/corr-' + name + '.pdf', orientation='landscape', format='pdf')
    plt.close()
