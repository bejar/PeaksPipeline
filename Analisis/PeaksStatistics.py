"""
.. module:: PeaksFilterRaw

PeaksFilterRaw
*************

:Description: PeaksCount

 Returns statistics about the peaks

:Authors: bejar
    

:Version: 

:Created on: 13/07/2015 8:37 

"""
import h5py
import numpy as np
from scipy.signal import butter, filtfilt

from Config.experiments import experiments, lexperiments
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd

__author__ = 'bejar'


def peaks_histogram(experiment):
    """
    Computes some statistics about the dada

    :param expname:
    :param iband:
    :param fband:
    :return:
    """

    f = experiment.open_experiment_data('r')
    print(experiment.dpath + experiment.name)

    dhisto = {}
    for s in experiment.sensors:
        dhisto[s] = []

    for df, ename in zip(experiment.datafiles, experiment.expnames):
        print(df, ename)
        for s in experiment.sensors:
            times = experiment.get_peaks_time(f, df, s)
            if times is not None:
                npk = times.shape[0]
                print(s, npk)
            else:
                print(s, 0)
                npk = 0
            dhisto[s].append(npk)

    for s in experiment.sensors:
        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        fig.set_figwidth(6)
        fig.set_figheight(4)
        ind = np.arange(len(dhisto[s]))
        fig.suptitle(experiment.name + '-' + s, fontsize=48)

        #print dhisto[s]
        for i, df in enumerate(experiment.datafiles):
            rects = ax.bar(ind[i], dhisto[s][i], 1, color= experiment.colors[i])
        #plt.show()
        fig.savefig(experiment.dpath + '/' + experiment.name + '/Results/peaks-count-' + experiment.name + '-' + s + '.pdf', orientation='landscape', format='pdf')
        plt.close()
    experiment.close_experiment_data(f)

def signal_distribution(experiment):
    """
    Histogram of the distribution of the signals

    :param experiment:
    :return:
    """

    f = experiment.open_experiment_data('r')
    print(experiment.dpath + experiment.name)

    for dfile, ename in zip(experiment.datafiles, experiment.expnames):
        print(dfile)
        data = experiment.get_raw_data(f, dfile)
        print data.shape
        data = pd.DataFrame(data[:,:len(experiment.sensors)], columns=experiment.sensors)
        plt.title(experiment.name + ' - ' + dfile)
        sns.boxplot(data=data, whis=6)
        #plt.show()
        plt.savefig(experiment.dpath + '/' + experiment.name + '/Results/signal-dist-' + experiment.name + '-' + dfile + '-' + ename + '.pdf', orientation='landscape', format='pdf')
        del data
        plt.close()

    experiment.close_experiment_data(f)

# ---------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")
    parser.add_argument('--hpeaks', help="Histogramas de los numeros de picos", action='store_true', default=False)
    parser.add_argument('--hsignal', help="Histogramas de la senyal cruda", action='store_true', default=False)

    args = parser.parse_args()
    lexperiments = args.exp
    hpeaks = args.hpeaks
    hsignal = args.hsignal

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511''e160204''e150514'
        lexperiments =['e130221']
        hpeaks = True
        hsignal = False



    for exp in lexperiments:

        if hpeaks:
            peaks_histogram(experiments[exp])
        if hsignal:
            signal_distribution(experiments[exp])
