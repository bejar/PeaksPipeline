"""
.. module:: PeakSyncroPloy

PeakSyncroPloy
*************

:Description: PeakSyncroPloy

    

:Authors: bejar
    

:Version: 

:Created on: 09/07/2015 15:53 

"""

__author__ = 'bejar'

from  matplotlib.backends.backend_pdf import PdfPages
from Sincronizaciones.PeaksSynchro import  compute_synchs
from Config.experiments import experiments
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import argparse
from sklearn.metrics import pairwise_distances_argmin_min

def plotSignals(signals,pp,n,m, title):
    fig = plt.figure()
    fig.set_figwidth(30)
    fig.set_figheight(16)
    #fig.suptitle(str(title), fontsize=48)
    i=1
    vmax = []
    vmin = []
    for s, _, _ in signals:
        vmax.append(np.max(s))
        vmin.append(np.min(s))

    for s,snm,v in signals:
        if min(s)!=max(s):
            plotSignalValues(fig,s,n,m,i,snm,v, np.max(vmax), np.min(vmin))
        else:
            plotDummy(fig,len(s),n,m,i,snm)
        i+=1
    fig.savefig(pp, orientation='landscape',format='pdf')
    plt.close()

#    plt.show()



# Plot a set of signals
def plotSignalValues(fig,signal1,n,m,p,name,v, maxaxis, minaxis):
    # minaxis=min(signal1)
    # maxaxis=max(signal1)
    num=len(signal1)
    sp1=fig.add_subplot(n,m,p)
    #plt.title(name)
    sp1.axis([0,num,minaxis,maxaxis])
    t = arange(0.0, num, 1)
    if v !=0:
        sp1.plot(t,signal1, 'r')
    else:
        sp1.plot(t,signal1, 'b')

#    plt.show()

def plotDummy(fig,num,n,m,p,name):
    minaxis=-1
    maxaxis=1
    sp1=fig.add_subplot(n,m,p)
    plt.title(name)
    sp1.axis([0,num,minaxis,maxaxis])
    t = arange(0.0, num, 1)
    sp1.plot(t,t)
#    plt.show()




# def plotSignalFile(name):
#     mats=scipy.io.loadmat( cpath+name+'-'+banda+'.mat')
#     data= mats['data']
#     chann=mats['names']
#     freq=40
#     off=60000
#     length=60000
#     for i in range(0,data.shape[0]+1,20):
#         lsignals=[]
#         for j in range(i,i+20):
#             if j<data.shape[0]:
#                 # copy the current segment
#                 orig=data[j][off:off+length]
#                 lsignals.append((orig,chann[j]))
#         plotSignals(lsignals,cres,10,2)


def draw_sincro(raw, lsync, num, nums, cres, name, sens):
    """
    Generates files with syncronizations
    """
    def busca_syn(syn, s):
        for sig, time, cl in syn:
            if s == sig:
                return sig, time
        return s, 0

    pp = PdfPages(cres+'/synch-raw' + name + '-' +str(num) + '-' + str(nums) + '.pdf')

    for i in range(num, nums):
        syn = lsync[i]
        ldraw = []
        for j in range(len(sens)):
            ldraw.append(busca_syn(syn, j))

        center = np.sum([v for s, v in ldraw if v > 0])/np.sum([1 for s, v in ldraw if v > 0])
        print i
        lsig = []
        for s, v in ldraw:
            lsig.append((raw[center-500:center+500, s], sens[s], v))

        plotSignals(lsig, pp, 2, 6, int(center))

    pp.close()

def compute_data_labels(fname, dfilec, dfile, sensor):
    """
    Computes the labels of the data using the centroids of the cluster in the file
    the labels are relabeled acording to the matching with the reference sensor

    Disabled the association using the Hungarian algorithm so the cluster index are
    the original ones

    :param dfile:
    :param sensor:
    :return:
    """
    f = h5py.File(datainfo.dpath + '/' + fname + '/' + fname + '.hdf5', 'r')

    d = f[dfilec + '/' + sensor + '/Clustering/' + 'Centers']
    centers = d[()]
    d = f[dfile + '/' + sensor + '/' + 'PeaksResamplePCA']
    data = d[()]
    f.close()

    labels, _ = pairwise_distances_argmin_min(data, centers)
    return labels



def select_sensor(synchs, sensor, slength):
    """
    Maintains only the syncs corresponding to the given sensor

    :param synchs:
    :param sensor:
    :return:
    """
    lres = []
    for syn in synchs:
        for s, _,_ in syn:
            if s == sensor and len(syn) >= slength:
                lres.append(syn)
    return lres



if __name__ == '__main__':
    window = 400
    print 'W=', int(round(window))
    # 'e150514''e120503''e110616''e150707''e151126''e120511'
    lexperiments = ['e110906o']

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()
    if args.exp:
        lexperiments = args.exp


    peakdata = {}
    for expname in lexperiments:

        datainfo = experiments[expname]
        rsensor = 'L6ri'
        nsensor = datainfo.sensors.index(rsensor)
        slength = 2
        filter = True

        # dfile = datainfo.datafiles[0]
        for dfile in [datainfo.datafiles[0]]:
            print dfile

            lsens_labels = []
            #compute the labels of the data
            for sensor in datainfo.sensors:
                lsens_labels.append(compute_data_labels( datainfo.name,
                                                        datainfo.datafiles[0], dfile, sensor))

            # Times of the peaks
            ltimes = []
            expcounts = []
            f = h5py.File(datainfo.dpath + '/' + datainfo.name + '/' + datainfo.name+ '.hdf5', 'r')
            if filter:
                ext = '-F400'
                d = f[dfile + '/RawFiltered']
            else:
                ext = ''
                d = f[dfile + '/Raw']

            raw = d[()]
            for sensor in datainfo.sensors:
                d = f[dfile + '/' + sensor + '/' + 'Time']
                data = d[()]
                #expcounts.append(data.shape[0])
                ltimes.append(data)
            f.close()
            lsynchs = compute_synchs(ltimes, lsens_labels, window=window)
            print len(lsynchs)
            lsynchs = select_sensor(lsynchs, nsensor, slength)
            print len(lsynchs)

            for i in range(0, 25, 100):
                draw_sincro(raw, lsynchs, i, i + 25, datainfo.dpath  + '/' + datainfo.name + '/Results',
                            dfile + '-' + rsensor + '-Len' + str(slength) + ext, datainfo.sensors)
