"""
.. module:: PeaksPCA

PeaksPCA
*************

:Description: PeaksPCA

    Performs a PCA of the resampled peaks and reconstructs them with only a number of them.
    After, it removes the mean a subwindow of the initial and final values of the signal
    Eventually saves the peaks in *Signal*/PeaksResamplePCA

    Uses joblib for paralelization

:Authors: bejar
    

:Version: 

:Created on: 26/03/2015 7:52

* 4/2/2016 - Adapting to changes in Experiment class

"""



import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

from Config.experiments import experiments
import argparse
from util.plots import show_two_signals
from itertools import product
import multiprocessing
from util.itertools import batchify
from util.Baseline import baseline_als, find_baseline
from Preproceso.Outliers import outliers_wavy
import time

__author__ = 'bejar'


def do_the_job(dfile, sensor, recenter=True, wtsel=None, clean=False, mbasal='meanfirst', alt_smooth=False, wavy=False, vpca=0.98):
    """
    Transforms the data reconstructing the peaks using some components of the PCA
    and uses the mean of the baseline points to move the peak

    :param pcap: Perform or not PCA
    :param dfile: datafile
    :param sensor: sensor
    :param components: Components selected from the PCA
    :param lbasal: Points to use to move the peak
    :param recenter: recenters the peak so it is in the center of the window
    :param basal: moving the peak so the begginning is closer to zero
                 'meanfirst', first n points
                 'meanmin', first min points of the first half of the peak
    :return:
    """
    print(datainfo.dpath + datainfo.name, sensor)

    f = datainfo.open_experiment_data(mode='r')
    data = datainfo.get_peaks_resample(f, dfile, sensor)
    datainfo.close_experiment_data(f)
    pcap = datainfo.get_peaks_smooth_parameters('pcasmooth')
    components = datainfo.get_peaks_smooth_parameters('components')
    baseline = datainfo.get_peaks_smooth_parameters('wbaseline')
    lbasal = range(baseline)
    if alt_smooth:
        parl = datainfo.get_peaks_alt_smooth_parameters('lambda')
        parp = datainfo.get_peaks_alt_smooth_parameters('p')

    if data is not None:
        # if there is a clean list of peaks then the PCA is computed only for the clean peaks
        if clean:
            lt = datainfo.get_clean_time(f, dfile, sensor)
            if lt is not None:
                ltime = list(lt[()])
                print(data.shape)
                data = data[ltime]
                print(data.shape)

        if alt_smooth:
            trans = np.zeros((data.shape[0], data.shape[1]))
            for i in range(data.shape[0]):
                trans[i] = baseline_als(data[i,:], parl, parp)
        elif pcap:
            pca = PCA(n_components=data.shape[1])
            res = pca.fit_transform(data)

            sexp = 0.0
            ncomp = 0
            while sexp < vpca:
                sexp += pca.explained_variance_ratio_[ncomp]
                ncomp += 1
            components = ncomp
            print('VEX=', np.sum(pca.explained_variance_ratio_[0:components]), components)
            res[:, components:] = 0
            trans = pca.inverse_transform(res)
        else:
            trans = data

        # If recenter, find the new center of the peak and crop the data to wtsel milliseconds
        if recenter:
            # Original window size in milliseconds
            wtsel_orig = datainfo.get_peaks_resample_parameters('wtsel')
            # current window midpoint
            midpoint = int(trans.shape[1]/2.0)
            # New window size
            wtlen = int(trans.shape[1]*(wtsel/wtsel_orig))
            wtdisc = int((trans.shape[1] - wtlen)/2.0)
            # in case we have a odd number of points in the window
            if wtlen + (2*wtdisc) != wtlen:
                wtdisci = wtdisc + 1
            else:
                wtdisci = wtdisc

            new_trans = np.zeros((trans.shape[0], wtlen))
            for pk in range(trans.shape[0]):
                # find current maximum around the midpoint of the current window
                # Fixed to 10 points around the center
                center = np.argmax(trans[pk, midpoint-10:midpoint+10])
                new_trans[pk] = trans[pk,wtdisci:wtlen-wtdisc]

            trans = new_trans

        # Substract the basal

        if mbasal == 'meanfirst':
            for row in range(trans.shape[0]):
                vals = trans[row, lbasal]
                basal = np.mean(vals)
                trans[row] -= basal
                #show_two_signals(trans[row]+basal, trans[row])
        elif mbasal == 'meanmin':
             for row in range(trans.shape[0]):
                vals = trans[row, 0:trans.shape[1]/2]
                vals = np.array(sorted(list(vals)))
                basal = np.mean(vals[lbasal])
                trans[row] -= basal
                #show_two_signals(trans[row]+basal, trans[row])
        elif mbasal == 'meanlast':
             for row in range(trans.shape[0]):

                vals = trans[row, (trans.shape[1]/3)*2:trans.shape[1]]
                basal = np.mean(vals)

                trans[row] -= basal
                #show_two_signals(trans[row]+basal, trans[row])
        elif mbasal == 'alternative':
             for row in range(trans.shape[0]):
                basal = find_baseline(trans[row, 0:trans.shape[1]/2], resolution=25)
                trans[row] -= basal

        if wavy:
            sel = outliers_wavy(trans)
            trans = trans[sel]

        return trans, dfile, sensor
    else:
        return None, dfile, sensor


# ---------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")
    parser.add_argument('--basal', default='meanfirst', help="Nombre de los experimentos")
    parser.add_argument('--altsmooth', help="Alternative smoothing", action='store_true', default=False)
    parser.add_argument('--wavy', help="Cuts too wavy signals", action='store_true', default=False)
    parser.add_argument('--pca', help="Percentage of variance preserved by PCA", default=0.98)
    parser.add_argument('--extra', help="Procesa sensores extra del experimento", action='store_true', default=False)

    args = parser.parse_args()
    lexperiments = args.exp

    mbasal = args.basal
    njobs = multiprocessing.cpu_count()

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511'
        lexperiments = ['e130221e2']
        mbasal =  'meanfirst' # 'alternative'
        altsmooth = False
        args.wavy = False
        args.extra = False
        args.pca = 0.98

    print('Begin Smoothing: ', time.ctime())
    for expname in lexperiments:
        datainfo = experiments[expname]

        if not args.extra:
            lsensors = datainfo.sensors
        else:
            lsensors = datainfo.extrasensors

        batches = batchify([i for i in product(datainfo.datafiles, lsensors)], njobs)

        if 'recenter' in datainfo.peaks_smooth:
            # If recenter is true a subwindow of the data has to be indicated to be able to re-crop the signal
            recenter = datainfo.peaks_smooth['recenter']
            wtsel = datainfo.peaks_smooth['wtsel']
        else:
            recenter = False
            wtsel = None


        for batch in batches:
            # Paralelize PCA computation
            res = Parallel(n_jobs=-1)(
                    delayed(do_the_job)(dfile, sensor, recenter=False, wtsel=None, clean=False, mbasal=mbasal, alt_smooth=altsmooth, wavy=args.wavy, vpca= args.pca) for dfile, sensor in batch)

            # Save all the data
            f = datainfo.open_experiment_data(mode='r+')
            for trans, dfile, sensor in res:
                if trans is not None:
                    print(dfile + '/' + sensor + '/' + 'PeaksResamplePCA',  time.ctime())
                    datainfo.save_peaks_resample_PCA(f, dfile, sensor, trans)
                    # if recenter:
                    #     f[dfile + '/' + sensor + '/PeaksResamplePCA'].attrs['baseline'] = recenter
                    #     f[dfile + '/' + sensor + '/PeaksResamplePCA'].attrs['wtsel'] = wtsel

            datainfo.close_experiment_data(f)
    print('End Smoothing: ', time.ctime())
