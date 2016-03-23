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

lexperiments = ['e150514']

expname = lexperiments[0]

datainfo = experiments[expname]
print(datainfo.dpath + datainfo.name + '/' + datainfo.name)
f = h5py.File(datainfo.dpath + datainfo.name + '/' + datainfo.name + '.hdf5', 'r')
print(expname)

for s in datainfo.sensors:
    print(s)
    ldatap = []
    ldatappca = []
    ltimes = []
    for dfiles in [datainfo.datafiles[0]]:
        print(dfiles)
        #d = f[dfiles + '/' + s + '/' + 'Peaks']
        #dataf = d[()]
        #ldatap.append(dataf)
        #d = f[dfiles + '/' + s + '/' + 'Time']
        #times = d[()]
        #ltimes.append(times)
        d = f[dfiles + '/' + s + '/' + 'PeaksResamplePCA']
        dataf = d[()]
        ldatappca.append(dataf)

    #data = ldatap[0] #np.concatenate(ldata)
    datapca = ldatappca[0] #np.concatenate(ldata)
    #ptime = ltimes[0]

    #print(len(data))
    for i in range(5): #range(data.shape[0]):
        # print dataraw[i]
        # print data[i]
        #print('T = %d'%ptime[i])
        #show_signal(data[i])
#        show_two_signals(data[i],datapca[i])
        show_signal(datapca[i])
