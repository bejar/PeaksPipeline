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
from util.plots import show_signal, plotSignals, show_two_signals, show_list_signals
from util.Baseline import baseline_als, find_baseline


from Config.experiments import experiments

lexperiments = ['e150514']

expname = lexperiments[0]

datainfo = experiments[expname]
print(datainfo.dpath + datainfo.name + '/' + datainfo.name)
f = datainfo.open_experiment_data(mode='r')

data = datainfo.get_peaks_resample(f, datainfo.datafiles[0], datainfo.sensors[11])
datapca = datainfo.get_peaks_resample_PCA(f, datainfo.datafiles[0], datainfo.sensors[11])

# for i in range(data.shape[0]):
#     dbase1 = baseline_als(data[i], 5, 0.9)
#     dbase2 = baseline_als(data[i], 10, 0.9)
#
#     show_list_signals([data[i], datapca[i], dbase1, dbase2], ['o', 'pca', 'rbase1', 'rbase2'])


for s in datainfo.sensors:
    print(s)
    ldatap = []
    ldatappca = []
    ltimes = []
    for dfiles in [datainfo.datafiles[0]]:
        print(dfiles)
        d = f[dfiles + '/' + s + '/' + 'PeaksResample']
        dataf = d[()]
        ldatap.append(dataf)
        #d = f[dfiles + '/' + s + '/' + 'Time']
        #times = d[()]
        #ltimes.append(times)
        d = f[dfiles + '/' + s + '/' + 'PeaksResamplePCA']
        dataf = d[()]
        ldatappca.append(dataf)

    data = ldatap[0] #np.concatenate(ldata)
    datapca = ldatappca[0] #np.concatenate(ldata)
    #ptime = ltimes[0]

    #print(len(data))
    long = data.shape[1]/3
    for i in range(5): #range(data.shape[0]):
        # print dataraw[i]
        # print data[i]
        #print('T = %d'%ptime[i])
        base = baseline_als(data[i], 5, 0.9)
        show_signal(base, find_baseline(data[i,:long], resolution=50))
        show_signal(base, find_baseline(base[i:long], resolution=100))
       #show_two_signals(data[i],datapca[i])
        # show_signal(datapca[i])



