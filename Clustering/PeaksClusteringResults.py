"""
.. module:: PeaksClustering

PeaksClustering
*************

:Description: PeaksClustering

    Genera los histogramas que corresponden con la probabilidad de los picos para la secuencia del experiment
    usando el clustering  global o el de control

:Authors: bejar
    

:Version: 

:Created on: 26/03/2015 8:10 

"""

from collections import Counter
from operator import itemgetter

import matplotlib.pyplot as plt
from pylab import *
import seaborn as sn
from sklearn.cluster import KMeans
from kemlglearn.cluster import KernelKMeans
from Config.experiments import experiments
from util.plots import plotSignals
import warnings
from util.distances import hellinger_distance
from util.misc import compute_centroids
import argparse
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")




__author__ = 'bejar'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")
    parser.add_argument('--globalclust', help="Use a global computed clustering", action='store_true', default=False)
    parser.add_argument('--hellinger', help="Show Hellinger distance", action='store_true', default=False)
    parser.add_argument('--extra', help="Procesa sensores extra del experimento", action='store_true', default=False)

    args = parser.parse_args()

    lexperiments = args.exp

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511''e150514''e110906o''e160802''e150514'
        args.hellinger = False
        args.globalclust = False
        lexperiments = ['e130221']
        args.extra = False

    if args.globalclust:
        ext='global'
    else:
        ext='ctrl'

    for expname in lexperiments:
        datainfo = experiments[expname]
        colors = datainfo.colors

        f = datainfo.open_experiment_data(mode='r')

        if not args.extra:
            lsensors = datainfo.sensors
            lclusters = datainfo.clusters
        else:
            lsensors = datainfo.extrasensors
            lclusters = datainfo.extraclusters

        for sensor, nclusters in zip(lsensors, lclusters):
            print(sensor)

            if args.globalclust:
                centroids = datainfo.get_peaks_global_clustering_centroids(f, sensor, nclusters)
            else:
                centroids = datainfo.get_peaks_clustering_centroids(f, datainfo.datafiles[0], sensor, nclusters)

            lsignals = []

            lhisto = []
            for dfile in datainfo.datafiles:

                labels = datainfo.compute_peaks_labels(f, dfile, sensor, nclusters, globalc=args.globalclust)

                histo = np.zeros(nclusters)
                for i in labels:
                    histo[i] += 1.0
                histo /= len(labels)
                lhisto.append(histo)

            # df = pd.DataFrame(np.array(lhisto), index=datainfo.expnames, columns=['class-%d'%i for i in range(1, nclusters+1)])
            #
            # if args.hellinger:
            #     lrms = []
            #     lhell = []
            #     for h, nphase in zip(lhisto, datainfo.expnames):
            #         rms = np.dot(lhisto[0] - h,  lhisto[0] - h)
            #         rms /= h.shape[0]
            #         lhell.append(hellinger_distance(h, lhisto[0]))
            #         lrms.append(np.sqrt(rms))
            #     df['Hellinger'] = lhell
            #     df['RMS'] = lrms

            # rfile = open(datainfo.dpath + '/' + datainfo.name + '/Results/cluster-histo-' + dfile + '-' + sensor + '-' +
            #             str(nclusters) + '-' + ext + '.txt', 'w')
            # rfile.write(df.to_string(line_width=200))
            #
            # rfile.close()

            matplotlib.rcParams.update({'font.size': 30})
            fig = plt.figure()
            ax = fig.add_subplot(2, 1, 1)
            fig.set_figwidth(30)
            fig.set_figheight(20)
            ind = np.arange(nclusters)  # the x locations for the groups
            width = 1.0/(len(lhisto)+1)   # the width of the bars
            ax.set_xticks(ind+width)
            ax.set_xticklabels(['class %d'% (i+1) for i in ind])
            for i, h in enumerate(lhisto):
                rects = ax.bar(ind+(i*width), h, width, color=colors[i])
            fig.suptitle(datainfo.name + '-' + sensor+ '-' + ext, fontsize=48)

            minaxis = np.min(centroids)
            maxaxis = np.max(centroids)

            for nc in range(nclusters):
                ax2 = fig.add_subplot(2, nclusters, nc+nclusters+1)
                signal = centroids[nc]
                plt.title(' ( '+str(nc+1)+' )')
                t = np.arange(0.0, len(signal), 1)
                ax2.axis([0, len(signal), minaxis, maxaxis])
                ax2.plot(t,signal)
                plt.axhline(linewidth=1, color='r', y=0)
            fig.savefig(datainfo.dpath + '/' + datainfo.name + '/Results/' + datainfo.name + '-' + sensor + '-' +
                        str(nclusters) + '-' + ext + '-histo-sort.pdf', orientation='landscape', format='pdf')
        #    plt.show()

            print('*******************')
            for nc in range(nclusters):
                lsignals.append((centroids[nc], str(nc)))

            if nclusters % 2 == 0:
                part = nclusters /2
            else:
                part = (nclusters /2) + 1
            plotSignals(lsignals, part, 2, maxaxis, minaxis, datainfo.name + '-' + sensor + '-' +
                        str(nclusters)+ '-' + ext, datainfo.name + '-' + sensor+ '-' + ext, datainfo.dpath + '/' + datainfo.name + '/Results/')
        datainfo.close_experiment_data(f)