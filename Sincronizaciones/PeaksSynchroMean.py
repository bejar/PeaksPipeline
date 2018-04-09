"""
.. module:: PeaksSynchroMean

PeaksSynchroMean
*************

:Description: PeaksSynchroMean

  Average plots of synchronized peaks

:Authors: bejar
    

:Version: 

:Created on: 04/04/2018 14:41 

"""

from Sincronizaciones.PeaksSynchro import compute_synchs
from Config.experiments import experiments
import h5py
import argparse
import numpy as np
from util.plots import plotListSignals

def contains_sensor(sync, sensor):
    """
    Returns if a synchronizations contains a specific sensor

    :param sensor:
    :return:
    """
    for s in sync:
        if s[0] == sensor:
            return True

    return False

def class_sync_sensor(sync, sensor):
    """
    PRE: The sensor must be in the synchronization list

    Returns the class of the peak of a sensor

    :param sensor:
    :return:
    """
    for s in sync:
        if s[0] == sensor:
            return s[2]

def time_sync_sensor(sync, sensor):
    """
    PRE: The sensor must be in the synchronization list

    Returns the time of the peak of a sensor

    :param sensor:
    :return:
    """
    for s in sync:
        if s[0] == sensor:
            return s[1]




__author__ = 'bejar'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()
    lexperiments = args.exp

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511''e110906o'
        lexperiments = ['e150514']

    window = 40
    wlen = 500

    peakdata = {}
    for expname in lexperiments:

        datainfo = experiments[expname]
        rsensor = 'L6ri'
        nsensor = datainfo.sensors.index(rsensor)

        for dfile, ename in zip(datainfo.datafiles, datainfo.expnames):
            print dfile, ename

            lsens_labels = []
            lsensors = datainfo.sensors
            lclusters = datainfo.clusters
            # compute the labels of the data
            f = datainfo.open_experiment_data(mode='r')
            for sensor in lsensors:
                nclusters = datainfo.clusters[datainfo.sensors.index(sensor)]
                lsens_labels.append(datainfo.compute_peaks_labels(f, dfile, sensor, nclusters, globalc=False))
            # Times of the peaks
            ltimes = []
            expcounts = []
            for sensor in lsensors:
                data = datainfo.get_peaks_time(f, dfile, sensor)
                expcounts.append(data.shape[0])
                ltimes.append(data)
            datainfo.close_experiment_data(f)
            lsynchs = compute_synchs(ltimes, lsens_labels, window=window, minlen=2)


            lsynchs_pruned = [trans for trans in lsynchs if contains_sensor(trans, nsensor)]
            vecsync = [[] for i in range(datainfo.clusters[nsensor])]

            print(len(lsynchs_pruned))

            for syn in lsynchs_pruned:
               vecsync[class_sync_sensor(syn,nsensor)].append(syn)

            f = datainfo.open_experiment_data(mode='r')
            datamat = datainfo.get_raw_data(f, dfile)

            for i in range(len(vecsync)):
                saverage = np.zeros((len(lsensors), (wlen*2)))
                scounts = np.zeros(len(lsensors))
                for syn in vecsync[i]:
                    stime = time_sync_sensor(syn, nsensor)

                    for j in range(len(lsensors)):
                        saverage[j] += datamat[stime-wlen:stime+wlen,j]

                    # for s in syn:
                    #     saverage[s[0]] += datamat[stime-wlen:stime+wlen,s[0]]
                    #     scounts[s[0]] += 1

                for i in range(len(lsensors)):
                    saverage[i] /= len(vecsync[i])

                # for i in range(len(lsensors)):
                #     saverage[i] /= scounts[i]

                plotListSignals(saverage, ncols=2)

            datainfo.close_experiment_data(f)



            # for s in lsynchs_pruned:
            #     print s

