"""
.. module:: Spectra

Spectra
*************

:Description: Spectra

    

:Authors: bejar
    

:Version: 

:Created on: 11/06/2015 11:38 

"""
from __future__ import division

__author__ = 'bejar'


import h5py

from pylab import *


from Config.experiments import experiments

import matplotlib.pyplot as plt

from util.STFT import stft
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()
    lexperiments = args.exp

    if not args.batch:
       # 'e120503''e110616''e150707''e151126''e120511''e150514''e110906o''e160204'
        lexperiments = ['e160204']


    peakdata = {}
    for expname in lexperiments:

        datainfo = experiments[expname]
        f = h5py.File(datainfo.dpath + datainfo.name + '/' + datainfo.name + '.hdf5', 'r')
        chunk = 2000000

        rate = datainfo.sampling
        for dfile in range(len(datainfo.datafiles)):
            d = f[datainfo.datafiles[dfile] + '/' + 'Raw']
            print d.shape, datainfo.sampling/2
            for s in range(len(datainfo.sensors)):
                print dfile, datainfo.sensors[s]
                length = int(rate)
                over = 2
                vec = stft(d[0:chunk, s], length, over)
                print vec.shape
                fig = plt.figure()
                fig.set_figwidth(100)
                fig.set_figheight(20)

                sp1 = fig.add_subplot(1, 1, 1)
                img = sp1.imshow(vec[:,0:100].T, cmap=plt.cm.seismic, interpolation='none')
                fig.savefig(datainfo.dpath + datainfo.name + '/' + '/Results/' + datainfo.datafiles[dfile] + '-'
                            + datainfo.sensors[s] + '-ftft.pdf', orientation='landscape', format='pdf')
                #plt.show()
                plt.close()
