"""
.. module:: PeaksSequenceSegmentation

PeaksSequenceSegmentation
*************

:Description: PeaksSequenceSegmentation

    

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

__author__ = 'bejar'




# --------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--globalclust', help="Use a global computed clustering", action='store_true', default=False)


    args = parser.parse_args()
    lexperiments = args.exp

    if not args.batch:
        lexperiments = ['e160317']
        args.globalclust = False


    for expname in lexperiments:

        datainfo = experiments[expname]
        lsensors = datainfo.sensors
        lclusters = datainfo.clusters

        for nclusters, sensor in zip(lclusters, lsensors):
            print(sensor)
            for dfile, ename in zip(datainfo.datafiles, datainfo.expnames):
                print(dfile, ename)
                d = datainfo.get_peaks_time(f, dfile, sensor)
                if d is not None:
                    clpeaks = datainfo.compute_peaks_labels(f, dfile, sensor, nclusters, globalc=args.globalclust)
                    timepeaks = d[()]

