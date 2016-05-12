"""
.. module:: PeaksClusteringValidate

PeaksClusteringValidate
*************

:Description: PeaksClusteringValidate

 Explores the possible number of clusters for the signals of using all the data of the phases of the experiments
using AMI stability


:Authors: bejar
    

:Version: 

:Created on: 06/05/2015 12:10 

"""

__author__ = 'bejar'

from numpy import mean, std
from sklearn import metrics
from sklearn.cluster import KMeans

from Config.experiments import experiments
from util.plots import show_signal, plotSignals
from collections import Counter
import numpy as np
import logging
import time
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")
    parser.add_argument('--nrepl', type=int, default=30, help="Numero de replicaciones")
    parser.add_argument('--rclust', nargs='+', type=int, default=[3, 20], help="Rango de numero de clusters")
    parser.add_argument('--globalclust', help="Use a global computed clustering", action='store_true', default=False)

    args = parser.parse_args()

    lexperiments = args.exp
    repl = args.nrepl
    rclust = range(args.rclust[0], args.rclust[1])

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511' 'e150707'
        lexperiments = ['e150514']
        args.globalclust = False
        repl = 40
        rclust = range(3, 22)

    itime = int(time.time())
    nchoice = 2

    niter = repl
    for expname in lexperiments:
        datainfo = experiments[expname]
        fname = datainfo.dpath + '/' + datainfo.name + '/Results/' + datainfo.name + '-val-%d.txt' % itime
        logging.basicConfig(filename=fname, filemode='w',
                            level=logging.INFO, format='%(message)s', datefmt='%m/%d/%Y %I:%M:%S')

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)

        logging.info('****************************')
        for sensor in datainfo.sensors:

            f = datainfo.open_experiment_data(mode='r')
            if args.globalclust:
                ldata = []
                for dfile in datainfo.datafiles:
                    data = datainfo.get_peaks_resample_PCA(f, dfile, sensor)
                    if data is not None:
                        idata = np.random.choice(range(data.shape[0]), data.shape[0] / nchoice, replace=False)
                        ldata.append(data[idata, :])

                data = np.vstack(ldata)
            else:
                data = datainfo.get_peaks_resample_PCA(f, datainfo.datafiles[0], sensor)

            datainfo.close_experiment_data(f)

            best = 0
            ncbest = 0
            logging.info('S= %s' % sensor)

            for nc in rclust:
                lclasif = []
                for i in range(niter):
                    k_means = KMeans(init='k-means++', n_clusters=nc, n_init=10, n_jobs=-1)
                    k_means.fit(data)
                    lclasif.append(k_means.labels_.copy())
                    # print '.',
                vnmi = []
                for i in range(niter):
                    for j in range(i + 1, niter):
                        nmi = metrics.adjusted_mutual_info_score(lclasif[i], lclasif[j])
                        vnmi.append(nmi)
                mn = mean(vnmi)
                if best < mn:
                    best = mn
                    ncbest = nc
                # print nc, mn
                logging.info('%d  %f %f' % (nc, mn, std(vnmi)))

            logging.info('S= %s NC= %d' % (sensor, ncbest))
            logging.info('****************************')
