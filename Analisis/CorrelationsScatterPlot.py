"""
.. module:: Correlations

Correlations
*************

:Description: Correlations

    Scatter plots of the cross-correlations of the sensors for all the experiment

:Authors: bejar
    

:Version: 

:Created on: 09/06/2015 8:29 

"""

__author__ = 'bejar'

import h5py
from pylab import *

from Config.experiments import experiments
import matplotlib.pyplot as plt

from operator import itemgetter

if __name__ == '__main__':
    lexperiments = ['e150514']

    peakdata = {}
    for expname in lexperiments:
        datainfo = experiments[expname]
        f = h5py.File(datainfo.dpath + datainfo.name + '/' + datainfo.name + '.hdf5', 'r+')

        # Correlation matrix for the first file of the experiment is used as reference
        d = np.array(f[datainfo.datafiles[0] + '/' + 'Raw'])
        corrmat = np.corrcoef(d, rowvar=0)
        ylabels = []
        for i, si in enumerate(datainfo.sensors):
            for j, sj in enumerate(datainfo.sensors):
                if i < j:
                    ylabels.append((si+'-'+sj, corrmat[i,j]))

        ylabels = sorted(ylabels, key=itemgetter(1))
        ylabels = [x for x, _ in ylabels]

        # Correlation matrices for all the experiment
        lcormat = []
        for ei in range(len(datainfo.datafiles)):
            d = np.array(f[datainfo.datafiles[ei] + '/' + 'Raw'])
            corrmat1 = np.corrcoef(d, rowvar=0)
            lcormat.append(corrmat1)

        # Scatterplot for each pair of correlation matrices
        for ei in range(len(datainfo.datafiles)):
            for ej in range(len(datainfo.datafiles)):
                if ei < ej:
                    corrmat1 = lcormat[ei]
                    corrmat2 = lcormat[ej]

                    fig, ax = plt.subplots(figsize=(20, 20))

                    dlabels1 = {}
                    dlabels2 = {}
                    for i, si in enumerate(datainfo.sensors):
                        for j, sj in enumerate(datainfo.sensors):
                            if i < j:
                                dlabels1[si + '-' + sj] = corrmat1[i, j]
                                dlabels2[si + '-' + sj] = corrmat2[i, j]
                    ydata1 = [dlabels1[x] for x in ylabels]
                    ydata2 = [dlabels2[x] for x in ylabels]

                    plt.xlabel('CrossCorrelation')
                    plt.title(datainfo.datafiles[ei] + '-' + datainfo.datafiles[ej])

                    plt.scatter(ydata1, ydata2)

                    plt.savefig(datainfo.dpath + datainfo.name + '/' + '/Results/' + datainfo.datafiles[ei] + '-'
                                + datainfo.datafiles[ej] + '-crosscorr.pdf', orientation='landscape', format='pdf')
                    plt.close()

