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

import h5py
from pylab import *

from Config.experiments import experiments
import matplotlib.pyplot as plt

from operator import itemgetter
import seaborn as sns
import argparse
import pandas as pd

__author__ = 'bejar'


def correlation_scatter_plot(datainfo, corrplot=False, barplot=False):
    """
    Generates the cross-correlation scatterplots of the experiment files

    :param datainfo:
    :return:
    """
    f = datainfo.open_experiment_data(mode='r')

    # Correlation matrix for the first file of the experiment is used as reference
    d = datainfo.get_raw_data(f, datainfo.datafiles[0])
    corrmat = np.corrcoef(d, rowvar=0)
    ylabels = []
    for i, si in enumerate(datainfo.sensors):
        for j, sj in enumerate(datainfo.sensors):
            if i < j:
                ylabels.append((si + '-' + sj, corrmat[i, j]))

    ylabels = sorted(ylabels, key=itemgetter(1))
    ylabels = [x for x, _ in ylabels]

    # Correlation matrices for all the experiment
    lcormat = []
    for ei in range(len(datainfo.datafiles)):
        d = datainfo.get_raw_data(f, datainfo.datafiles[ei])
        corrmat1 = np.corrcoef(d, rowvar=0)
        lcormat.append(corrmat1)

    matplotlib.rcParams.update({'font.size': 12})
    if barplot:
        for ei, dfile in enumerate(datainfo.datafiles):
            cmat = lcormat[ei]
            dlabels = {}
            for i, si in enumerate(datainfo.sensors):
                for j, sj in enumerate(datainfo.sensors):
                    if i < j:
                        dlabels[si + '-' + sj] = cmat[i, j]
            ydata = [dlabels[x] for x in ylabels]

            data = pd.DataFrame({'corr': ydata, 'sens': ylabels})

            plt.title(dfile)
            ax = sns.barplot(x="corr", y="sens", data=data, orient='h', palette="Blues_d")
            plt.savefig(datainfo.dpath + datainfo.name + '/' + '/Results/corrbarplot-' + dfile +
                        '.pdf', orientation='landscape', format='pdf')
            plt.close()

    if corrplot:
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

                    plt.savefig(
                        datainfo.dpath + datainfo.name + '/' + '/Results/crosscorr-' + datainfo.datafiles[ei] + '-' +
                        datainfo.datafiles[ej] + '.pdf', orientation='landscape', format='pdf')
                    plt.close()

    datainfo.close_experiment_data(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")
    parser.add_argument('--corrplot', help="Crosscorrelation plots", action='store_true', default=False)
    parser.add_argument('--barplot', help="Correlation bar plots", action='store_true', default=False)

    args = parser.parse_args()
    lexperiments = args.exp

    if not args.batch:
        # 'e120503''e110616''e150707''e151126''e120511''e150514''e110906o'
        lexperiments = ['e150514']
        args.corrplot = False
        args.barplot = True

    peakdata = {}
    for expname in lexperiments:
        datainfo = experiments[expname]
        correlation_scatter_plot(datainfo, corrplot=args.corrplot, barplot=args.barplot)
