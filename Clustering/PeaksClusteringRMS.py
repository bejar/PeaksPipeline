"""
.. module:: PeaksClusteringRMS

PeaksClusteringRMS
*************

:Description: PeaksClusteringRMS

    

:Authors: bejar
    

:Version: 

:Created on: 26/06/2019 12:26 

"""

__author__ = 'bejar'
from collections import Counter
from operator import itemgetter

import matplotlib.pyplot as plt
#from pylab import *
import seaborn as sn
from sklearn.cluster import KMeans
from kemlglearn.cluster import KernelKMeans
from Config.experiments import experiments
# from util.plots import plotSignals
import warnings
from util.distances import hellinger_distance
# from util.misc import compute_centroids
import argparse
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")
from matplotlib import cm as cm
__author__ = 'bejar'


def RMS(lhisto, lfiles):
    """
    Compute RMS significance for histograms

    :param lhisto:
    :param lfiles:
    :return:
    """
    drms = {}
    for i, (histo1, file1) in enumerate(zip(lhisto, lfiles)):
        arr = np.zeros(len(lhisto))
        for j, (histo2, file2) in enumerate(zip(lhisto, lfiles)):
            if i!=j:
                sh1 = np.sum(histo1)
                sh2 = np.sum(histo2)
                k=sh1/sh2
                lsm = []
                for b1, b2 in zip(histo1, histo2):
                    v = (b1 - (k * b2)) /np.sqrt((b1 + (k*k * b2)))
                    lsm.append(v)

                msm = np.mean(lsm)
                rms = (np.array(lsm) - msm)
                rms = rms **2
                rms = np.sqrt(np.sum(rms)/ len(histo1))
                arr[j] = rms
            else:
                arr[j]=0
        drms[file1] = arr
    return drms



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()

    lexperiments = args.exp

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511''e110906o''e160802'
        lexperiments = ['e130221'] #'e130221''e150514'


    for expname in lexperiments:
        datainfo = experiments[expname]
        colors = datainfo.colors

        f = datainfo.open_experiment_data(mode='r')


        lsensors = datainfo.sensors
        lclusters = datainfo.clusters


        for sensor, nclusters in zip(lsensors, lclusters):
            print(sensor)


            centroids = datainfo.get_peaks_clustering_centroids(f, datainfo.datafiles[0], sensor, nclusters)

            lsignals = []

            lhisto = []
            for dfile in datainfo.datafiles:

                labels = datainfo.compute_peaks_labels(f, dfile, sensor, nclusters)

                histo = np.zeros(nclusters)
                for i in labels:
                    histo[i] += 1.0
                # print(histo)
                lhisto.append(histo)

            dres = RMS(lhisto, datainfo.expnames)

            df = pd.DataFrame(dres)
            df.index=datainfo.expnames
            # print(df)

            fig = plt.figure()
            tlabels = ['ctrl1']

            for i in range(len(datainfo.expnames[1:])):
                nexp1 = datainfo.expnames[i+1]
                nexp2 = datainfo.expnames[i]
                if nexp1[:2] ==  nexp2[:2]:
                    tlabels.append('')
                else:
                    tlabels.append(datainfo.expnames[i+1])


            sn.heatmap(df,cmap='viridis',
                xticklabels=tlabels,
                yticklabels=tlabels, cbar_kws={'ticks':[0,1,3]}
                       )
            # plt.show()
            fig.savefig(datainfo.dpath + '/' + datainfo.name + '/Results/' + datainfo.name + '-' + sensor + '-' +
                        str(nclusters) +  '-RMS.pdf', orientation='landscape', format='pdf')
