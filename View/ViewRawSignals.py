"""
.. module:: ViewPeaks

ViewPeaks
*************

:Description: ViewPeaks

    

:Authors: bejar
    

:Version: 

:Created on: 27/05/2015 14:50 

"""

__author__ = 'bejar'

import h5py
from util.plots import show_signal, plotSignals, show_two_signals


from Config.experiments import experiments

from pylab import *

from scipy.signal import butter, filtfilt

#'e120503'

lexperiments = ['e110616']
expname = lexperiments[0]

datainfo = experiments[expname]

f = h5py.File(datainfo.dpath + datainfo.name + '/'+ datainfo.name + '.hdf5', 'r')

nfile = 0
nsensor = 8
tinit = 0
tfin = 600000

dfile = datainfo.datafiles[nfile]


print(dfile)
print(datainfo.sensors[nsensor])


d = f[dfile + '/' + 'Raw']
samp = f[dfile + '/Raw'].attrs['Sampling']
data = d[()]

for i in range(0, d.shape[0], 500000):
    print(i)

    show_signal(data[i:i+500000, nsensor])
