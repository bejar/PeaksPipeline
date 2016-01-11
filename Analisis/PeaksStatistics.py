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
__author__ = 'bejar'


def filter_data(experiment):
    """
    Filters and saves the raw signal in the datafile

    :param expname:
    :param iband:
    :param fband:
    :return:
    """


    f = h5py.File(experiment.dpath + experiment.name + '/' + experiment.name + '.hdf5', 'r+')
    print( experiment.dpath + experiment.name)



    dhisto = {}
    for s in experiment.sensors:
        dhisto[s] = []

    for i, df in enumerate(experiment.datafiles):
        print(df)
        for s in experiment.sensors:
            if df + '/' + s + '/Time' in f:
                times = f[df + '/' + s + '/Time']
                print(s, times.shape[0])
                if i >= 20:
                    npk = times.shape[0]
                else:
                    npk = times.shape[0] / 2.0
            else:
                print(s,0)
                npk = 0
            dhisto[s].append(npk)

    for s in experiment.sensors:
        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        fig.set_figwidth(60)
        fig.set_figheight(40)
        ind = np.arange(len(dhisto[s]))
        fig.suptitle(experiment.name + '-' + s, fontsize=48)

        #print dhisto[s]
        for i, df in enumerate(experiment.datafiles):
            rects = ax.bar(ind[i], dhisto[s][i], 1, color= experiment.colors[i])
        #plt.show()
        fig.savefig(experiment.dpath + '/' + experiment.name + '/Results/' + experiment.name + '-' + s + '-peaks-count.pdf', orientation='landscape', format='pdf')
        plt.close()

    f.close()


# ---------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    lexperiments = ['e151126']
    for exp in lexperiments:
        filter_data(experiments[exp])
