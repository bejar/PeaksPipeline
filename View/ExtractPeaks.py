"""
.. module:: ExtractPeaks

ExtractPeaks
*************

:Description: ExtractPeaks

    

:Authors: bejar
    

:Version: 

:Created on: 08/07/2016 9:11 

"""


import matplotlib.pyplot as plt
from pylab import *
from Config.experiments import experiments
from scipy.signal import resample
from sklearn.decomposition import PCA

__author__ = 'bejar'

if __name__ == '__main__':

    expname = 'e150514'
    npack = 100
    datainfo = experiments[expname]
    sensor = 'L6ri'
    nsensor = 6
    cluster = 10
    width = 1000
    nclusters = datainfo.clusters[nsensor]
    f = datainfo.open_experiment_data(mode='r')


    centroids = datainfo.get_peaks_clustering_centroids(f, datainfo.datafiles[0], sensor, nclusters)

    lsignals = []


    dfile = datainfo.datafiles[0]

    labels = datainfo.compute_peaks_labels(f, dfile, sensor, nclusters)
    times = datainfo.get_peaks_time(f, dfile, sensor)
    data = datainfo.get_raw_data(f, dfile)
    # Peaks from the clusters
    select = labels == cluster

    print(sum(select))

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    fig.set_figwidth(1)
    fig.set_figheight(npack+2)


    wtlen = int(((2*width)+1)/datainfo.peaks_resampling['rsfactor'])
    print(wtlen)
    vdata = np.zeros((sum(select), wtlen))
    npk = 0
    for idx, nc in enumerate(select):
        if nc:
            vdata[npk] = resample(data[times[idx]-width:times[idx]+width+1, nsensor], wtlen, axis=0, window=wtlen*2)
            npk += 1

    pca = PCA(n_components=vdata.shape[1])
    res = pca.fit_transform(vdata)

    sexp = 0.0
    ncomp = 0
    while sexp < 0.98:
        sexp += pca.explained_variance_ratio_[ncomp]
        ncomp += 1
    components = ncomp
    print('VEX=', np.sum(pca.explained_variance_ratio_[0:components]), components)
    res[:, components:] = 0

    vdata = pca.inverse_transform(res)

    minaxis = 0
    maxaxis = npack + 2
    ax.axis([0, wtlen, minaxis, maxaxis])
    t = arange(0.0, wtlen, 1)

    pack = 0
    npk = 1
    for idx, nc in enumerate(range(sum(select))):
        ax.plot(t, vdata[idx]+(npk-1), color='k')
        plt.axhline(linewidth=1, color='r', y=npk-1)
        npk += 1
        if npk % npack == 0:
            fig.savefig(datainfo.dpath + '/' + datainfo.name + '/Results/' + datainfo.name + '-' + sensor + '-' +
                        str(nclusters) + '-' + str(cluster+1) + '-' + str(pack)+ '-cluster-peaks.svg', orientation='landscape', format='svg')
            plt.close()
            fig = plt.figure()

            ax = fig.add_subplot(1, 1, 1)
            ax.axis([0, wtlen, minaxis, maxaxis])
            fig.set_figwidth(1)
            fig.set_figheight(npack + 2)

            npk = 1
            pack += 1
            print(pack)

    # For the last batch of peaks
    if npk % npack != 0:
        fig.savefig(datainfo.dpath + '/' + datainfo.name + '/Results/' + datainfo.name + '-' + sensor + '-' +
                    str(nclusters) + '-' + str(cluster+1) + '-' + str(pack)+ '-cluster-peaks.svg', orientation='landscape', format='svg')
        plt.close()
        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        ax.axis([0, wtlen, minaxis, maxaxis])
        fig.set_figwidth(1)
        fig.set_figheight(npack + 2)




    datainfo.close_experiment_data(f)