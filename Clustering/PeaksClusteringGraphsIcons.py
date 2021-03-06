"""
.. module:: ComputeIcons

ComputeIcons
*************

:Description: ComputeIcons

    Generates the icons for the sequential analysis graph

:Authors: bejar
    

:Version: 

:Created on: 20/10/2014 9:09 

"""

__author__ = 'bejar'

import scipy.io
from pylab import *
import h5py
import numpy as np
import os

from Config.experiments import experiments
import argparse


def plotSignalValues(signals, dfile, sensor, ncl, nc, globalclust=False):

    ext = ''
    if globalclust:
        ext = '.g'
    fig = plt.figure()
    minaxis = -0.1
    maxaxis = 0.3
    sp1 = fig.add_subplot(1, 1, 1)
    sp1.axis([0, peakLength, minaxis, maxaxis])

    sp1.axes.get_xaxis().set_visible(False)
    sp1.axes.get_yaxis().set_visible(False)

    t = arange(0.0, peakLength, 1)
    sp1.plot(t, signals, 'b')
    plt.axhline(linewidth=1, color='r', y=0)
    fig.set_figwidth(1.5)
    fig.set_figheight(1.5)

    # plt.show()
    fig.savefig(datainfo.dpath + '/' + datainfo.name +'/Results/icons/' + dfile + sensor + '.nc' + str(ncl) + '.cl' + str(nc) + ext + '.pdf', orientation='landscape', format='pdf',
                pad_inches=0.1)
    fig.savefig(datainfo.dpath + '/' + datainfo.name +'/Results/icons/' + dfile + sensor + '.nc' + str(ncl) + '.cl' + str(nc) + ext + '.png', orientation='landscape', format='png',
                pad_inches=0.1)
    fig.savefig(datainfo.dpath + '/' + datainfo.name +'/Results/icons/' + dfile + sensor + '.nc' + str(ncl) + '.cl' + str(nc) + ext + '.jpg', orientation='landscape', format='jpeg',
                pad_inches=0.1)
    plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")
    parser.add_argument('--globalclust', help="Use a global computed clustering", action='store_true', default=False)

    args = parser.parse_args()
    lexperiments = args.exp

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511''e110906o''e161201'
        lexperiments = ['e120511']
        args.globalclust = False

    peakdata = {}
    for expname in lexperiments:

        datainfo = experiments[expname]

        f = datainfo.open_experiment_data(mode='r')
        if not os.path.exists(datainfo.dpath + '/' + datainfo.name + '/Results/icons'):
            os.makedirs(datainfo.dpath + '/' + datainfo.name + '/Results/icons')


        dfile = datainfo.datafiles[0]
        for sensor, ncl in zip(datainfo.sensors, datainfo.clusters):
            if args.globalclust:
                centers = datainfo.get_peaks_global_clustering_centroids(f, sensor, ncl)
            else:
                centers = datainfo.get_peaks_clustering_centroids(f, dfile, sensor, ncl)
            peakLength = centers.shape[1]
            for i in range(centers.shape[0]):
                plotSignalValues(centers[i], expname, sensor, ncl, i + 1, globalclust=args.globalclust)
        datainfo.close_experiment_data(f)
