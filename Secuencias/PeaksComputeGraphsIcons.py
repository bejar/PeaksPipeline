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

def plotSignalValues(signals, dfile, sensor, nc):
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
    fig.savefig(datainfo.dpath + '/' + datainfo.name +'/Results/icons/' + dfile + sensor + '.cl' + str(nc) + '.png', orientation='landscape', format='png',
                pad_inches=0.1)
    plt.close()




if __name__ == '__main__':
    # 'e150514''e120503''e110616''e150707''e151126''e120511'
    lexperiments = ['e150707', 'e150514']

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()
    if args.exp:
        lexperiments = args.exp

    peakdata = {}
    for expname in lexperiments:

        datainfo = experiments[expname]
        f = h5py.File(datainfo.dpath + '/' + datainfo.name + '/' + datainfo.name + '.hdf5', 'r')
        if not os.path.exists(datainfo.dpath + '/' + datainfo.name + '/Results/icons'):
            os.makedirs(datainfo.dpath + '/' + datainfo.name + '/Results/icons')


        # dfile = datainfo.datafiles[0]
        for dfile in [datainfo.datafiles[0]]:
            print(dfile)

            lsens_labels = []
            #compute the labels of the data
            for sensor in datainfo.sensors:

                d = f[dfile + '/' + sensor + '/Clustering/' + 'Centers']
                centers = d[()]
                peakLength = centers.shape[1]
                for i in range(centers.shape[0]):
                    plotSignalValues(centers[i], expname, sensor, i + 1)
        f.close()
