# coding=utf-8
"""
.. module:: cdp_identification

cdp_identification
*************

:Description: cdp_identification

    Port to python of the matlab code for CDP identification with parallelization

    Identificacion de los picos en los sensores, version con paralelismo

    Lanza en paralelo tantos procesos de identificacion de picos como cores tenga la maquina, gestionado automaticamente
    por la libreria joblib

    Los parametros de identificacion de los picos se definen en el objeto Experiment
      * wtime = Tamaño de la ventana a usar en la identificacion
      * low = frecuencia minima a usar en la FFT
      * high = frecuencia maxima a usar en la FFT
      * threshold = amplitud minima del pico
:Authors: bejar
    

:Version: 

:Created on: 30/06/2015 15:19 

"""

__author__ = 'bejar'

import time

import h5py
import numpy as np
from joblib import Parallel, delayed

from Config.experiments import experiments
from Config.experiments import lexperiments


def uniquetol(peaks, tol):
    """
    returns the list of the indices of the peaks that are at a distance less than the tolerance to the
    previous peaks

    The list of peaks is a triple, the time of the peak is the first element

    :param peaks:
    :param tol:
    :return:
    """

    if len(peaks) != 0:
        lres = []
        curr = peaks[0]
        lres.append(0)
        for i in range(1, peaks.shape[0]):
            if peaks[i] - curr > tol:
                curr = peaks[i]
                lres.append(i)
        return lres
    else:
        return []


def integIneqG(data, QI1, QI2, QF1, QF2):
    """
    Pseudo peak test, compares the sum of the bins of the data to check if
    it has a peaky shape

    It should be generalize to any number of bins, currently only 7 bins are possible

    :param QF2:
    :param QF1:
    :param QI2:
    :param QI1:
    :param data:
    :return:
    """
    bins = 7
    # bins=8;

    dataLength = len(data)

    minval = np.min(data)
    maxval = np.max(data)

    normData = (data - minval) / (maxval - minval)

    binLength = dataLength / bins

    integData = np.zeros(bins)
    for i in range(bins):
        integData[i] = np.sum(normData[i * binLength:(i + 1) * binLength])

    sumData = np.sum(integData)

    quotI1 = integData[2] / sumData
    quotI2 = (integData[2] + integData[3]) / sumData
    quotF1 = integData[7] / sumData
    quotF2 = (integData[6] + integData[7]) / sumData

    intTest = (integData[2] + integData[3] < integData[4] + integData[5]) and \
              (integData[3] < integData[4]) and \
              (integData[5] + integData[6]) and \
              (integData[6] + integData[7] < integData[4] + integData[5]) and \
              (quotI1 < QI1) and (quotI2 < QI2) and (quotF1 < QF1) and (quotF2 < QF2)

    return intTest


def fifft(Nfft, fmask, peak, dw, N):
    """
    Returns the peaks with filtered frequencies

    Based on the original Matlab code

    Tappering disabled for now

    :param N:
    :param dw:
    :param fmask:
    :param Nfft:
    :param peak:
    :return:
    """

    xo = np.zeros(Nfft)
    xo[dw:N + dw] = peak

    y = np.fft.rfft(xo) / N  # perform fft transform

    # filter amplitudes (deletes the aplitudes outside the cutoff ranges)
    y2 = ffft(fmask, y)

    X = np.fft.irfft(y2)
    X = X[dw:N + dw - 1] * N
    return X, y, y2


def ffft(fmask, y):
    """
    Suposedly deletes the amplitudes of the fft that are outside the cutoffs so
    the inverse FFT returns the smoothed peak

    :param f:
    :param y:
    :param par1:
    :param par2:
    :param myfreq:
    :return:
    """
    nf = len(fmask)
    ny = len(y)

    y2 = np.zeros(ny, dtype=np.complex)
    # cutoff filter, computes the indices of the frequencies among the cutoffs
    if fmask is not None:
        y2[fmask] = y[fmask]  # insert required elements
    else:
        y2 = y

    # create a conjugate symmetric vector of amplitudes
    for k in range(nf + 1, ny):
        y2[k] = np.conj(y2[((ny - k) % ny)])
    return y2


def cdp_identification(X, wtime, datainfo, sensor, ifreq=0.0, ffreq=200, threshold=0.05):
    """
    Identification of peaks

    :param X: Time series data
    :param wtime: Window length
    :param Fs: Sampling
    :return:
    """

    print('Sensor: ', sensor, time.ctime())

    Fs = datainfo.sampling

    tapering = 0.0  # tapering window use tukey window tap=0 no window, tap=1 max tapering
    fft_freq = None  # max number of freqs used in FFT

    upthreshold = 1  # Outliers threshold *** ADDED by Javier
    downthreshold = -0.4  # Outliers threshold *** ADDED by Javier

    peakprecision = wtime / 12  # Peaks localization time resolution in points
    RCoinc = wtime / 6  # Peak synchronization radius
    Tpk = wtime / 4  # Quality filter time window subdivision
    qualc = True  # apply quality cut, =0 do not apply quality cut
    factp = 1.5  # Quality cut on Peaks in windows
    factm = 1.5  # Pv=max(FFT(signal)) Dm=mean(FFT(signal before Pv)(1:Tpk))
    # Dp=mean(FFT(signal after Pv))(Tw-Tpk:Tw)
    # quality cut:  Pv>factm*Dm&&Pv>factp*Dp
    quot = [0.19, 0.36, 0.23, 0.36]  # Integration test parameters
    testsn = True  # =1 apply signal to noise ratio quality cut =0 do not apply this quality cut

    forceTm = 10  # =0 smart peak search =Tm force fixed Tm step peak search
    forcedT = 1  # percentage of the data to be processed

    freq = fft_freq
    if freq == 0:
        freq = None
    ismot = 2  # =1 or 2 use FFT to smooth signal, =3 use wavelet to smooth signal

    # Peaks location in time window =Tw/npz
    npz = 2  # centered peak

    Nmxx = X.shape[0]

    Nmax = np.floor(forcedT * Nmxx)  # Max numner of points in file
    Tmax = Nmax / Fs  # Time max analyzed in sec
    Tw = int(2 * np.round(wtime * Fs / 2))  # Number of points in window (Tw must be an even number)
    t = np.array(range(Tw)) / Fs  # Time axis in window

    Nfft = int(2 ** np.ceil(np.log2(Tw)))

    f = Fs / 2 * np.linspace(0, 1, 1 + Nfft / 2)  # create freqs vector

    if ifreq is not None or ffreq is not None:
        ind1 = f <= ffreq
        ind2 = f >= ifreq
        fmask = np.logical_and(ind2, ind1)

    Tpk = int(np.floor(Tpk * Fs))
    peakprecision = np.floor(peakprecision * Fs)

    ipeakM = None  # This list contains the index of the center of the peak, the Sum and the RMS
    SNp = None
    RMSp = None
    Tm = 1
    dw = np.floor((Nfft - Tw) / 2)

    tstop = 0
    tstart = Tw
    ipeakMj = []
    SNpj = []
    RMSpj = []
    nt = 0
    while tstop < Nmax - Tw:
        tstop = min(Nmax, tstart + Tw - 1)
        xs = X[tstart:tstop]
        Nl = len(xs)
        if Nl < Tw:
            xs = np.hstack((xs, np.zeros(Tw - Nl)))

        xf, y, y2 = fifft(Nfft, fmask, xs, dw, Tw)  # signal smooth in frequency interval

        xf -= np.min(xf)
        qpeak = (np.max(xf) > threshold) and (np.max(xf) < upthreshold) and (
        np.min(xf) > downthreshold)  # *** up/downthreshold ADDED by Javier

        # Peaks second level cuts we only consider time windows
        # with centered peaks and minimum peaks amplitude >threshold
        Tm = np.argmax(xf[np.floor(Tw / npz) + 2:Tw]) + 1  # smart peaks search
        if qpeak:  # store the time window only if there is a peak
            Pv = np.max(xf[np.floor(peakprecision / 4):Tw - np.floor(peakprecision / 4)])
            indp = np.argmax(xf[np.floor(peakprecision / 4):Tw - np.floor(peakprecision / 4)])
            Pkv = np.max(xf[np.floor(Tw / npz) - peakprecision:np.floor(Tw / npz) + peakprecision])

            if (Pv == Pkv):
                # evaluate quality of the peak
                Tqpeak = False
                if qualc:
                    Dp = 1
                    if Tpk > 0:
                        Dp = abs(np.mean(xf[0:Tpk]))

                    Dm = 1
                    if Tpk > 0:
                        Dm = abs(np.mean(xf[Tw - Tpk:Tw]))
                    Tqpeak = (Pv > (factp * Dp)) and (Pv > (factm * Dm))

                    # Tqint = integIneqG(xf,quot)

                # check the peak
                if Tqpeak:
                    # Store the index of the peak, sum ratio and RMS
                    ipeakMj.append(tstart + np.floor(peakprecision / 4) + indp)
                    SNpj.append(np.sum(y2 * np.conj(y2)) / np.sum(y * np.conj(y)))
                    RMSpj.append(np.std(xs))
                    # store time of max peak in window
                    Tm = int(np.floor(Tw / npz) + 1)

                    # check which peaks are we throwing away
                    # noquality=Tcentr and not Tqpeak
                    # if noquality:
                    #     ipeakMnoQj.append(tstart+np.floor(peakprecision/4)+indp)

        if forceTm != 0:
            Tm = forceTm  # force exhaustive peaks search
        tstart += Tm
        nt += 1

    # Add the peaks of the signal to the datastructure
    ipeakM = np.array(ipeakMj)
    SNp = np.array(SNpj)
    RMSp = np.array(RMSpj)

    # print 'Filtering near peaks', sensor, time.ctime()
    # This eliminates all the peaks that are at a distance less than the peak precision parameter
    # TODO: Change the previous part so this is not necessary
    lind = uniquetol(ipeakM, peakprecision)
    ipeakMsel = ipeakM[lind]
    SNpsel = SNp[lind]
    RMSpsel = RMSp[lind]

    # for peaks in ipeakMsel:
    #     print len(peaks)

    # Signal to noise ratio filtering
    signal_noise_tolerance = 1.4  # Tolerance for Signal/Noise ratio
    co = 0.96  # parameters used in SNp selection cut thdPP=co*(1-ao*exp(-so*RMSp(i,j)))/(1-bo*exp(-so*RMSp(i,j)));
    ao = 2
    bo = 1
    so = 50
    ko = 6.41

    # print 'Filtering Noise Ratio Peaks ', sensor, time.ctime()


    # thdP = np.mean(SNpsel[i]) - signal_noise_tolerance * np.std(SNpsel[i])
    # thdR = np.mean(RMSpsel[i]) + ko * np.std(RMSpsel[j])
    ipeakMj = []
    for j in range(ipeakMsel.shape[0]):
        # tcenter = ipeakMsel[sensor][j]
        # tstart = np.max([1, tcenter - np.floor(Tw / npz)])
        # tstop = np.min([Nmax, tstart + Tw - 1])
        # tmp = X[tstart:tstop, i]
        #
        # Nl = len(tmp)
        # if Nl < Tw:
        #     tmp = np.hstack((tmp, np.zeros(Tw-Nl)))

        # PeakM(i,:,j)=tmp; # select the signal
        thdPP = co * (1 - ao * np.exp(-so * RMSpsel[j])) / (1 - bo * np.exp(-so * RMSpsel[j]))
        if SNpsel[j] > thdPP:
            ipeakMj.append(ipeakMsel[j])
    ipeakM = np.array(ipeakMj)

    # for peaks in ipeakM:
    #     print len(peaks)

    return sensor, ipeakM


# ---------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # 'e150514'
    lexperiments = ['e120503']

    datasufix = ''  #

    # Preparado para procesar un conjunto de experimentos a la vez
    for expname in lexperiments:
        datainfo = experiments[expname]

        wtime = datainfo.peaks_id_params['wtime']  # Window length in miliseconds
        ifreq = datainfo.peaks_id_params['low']  # Frequency cutoff low
        ffreq = datainfo.peaks_id_params['high']  # Frequency cutoff high
        threshold = datainfo.peaks_id_params['threshold']  # Peaks Max-Min in window above threshold in amplitude

        print(wtime, ifreq, ffreq, threshold)

        sampling = datainfo.sampling  # / 6.0
        Tw = int(2 * np.round(wtime * sampling / 2))
        f = h5py.File(datainfo.dpath  + datainfo.name + '/' + datainfo.name + '.hdf5', 'r+')

        for dfile in datainfo.datafiles:
            print(dfile)
            d = f[dfile + '/Raw']

            raw = d[()]
            print('Peaks identification: ', time.ctime())
            peaks = Parallel(n_jobs=-1)(
                delayed(cdp_identification)(raw[:, i], wtime, datainfo, s, ifreq=ifreq, ffreq=ffreq,
                                            threshold=threshold) for i, s in enumerate(datainfo.sensors))
            # peaks = cdp_identification(raw, wtime, datainfo)
            print('The end ', time.ctime())

            for s, p in peaks:
                print(s, len(p))
                if dfile + '/' + s in f:
                    del f[dfile + '/' + s]
                dgroup = f.create_group(dfile + '/' + s)
                # Time of the peak
                dgroup.create_dataset('Time', p.shape, dtype='i', data=p,
                                      compression='gzip')

                f[dfile + '/' + s + '/Time'].attrs['wtime'] = wtime
                f[dfile + '/' + s + '/Time'].attrs['low'] = ifreq
                f[dfile + '/' + s + '/Time'].attrs['high'] = ffreq
                f[dfile + '/' + s + '/Time'].attrs['threshold'] = threshold
                rawpeaks = np.zeros((p.shape[0], Tw))
                # Extraction of the window around the peak maximum

                i = datainfo.sensors.index(s)
                for j in range(p.shape[0]):
                    tstart = p[j] - np.floor(Tw / 2)
                    tstop = tstart + Tw
                    rawpeaks[j, :] = raw[tstart:tstop, i]

                # Peak Data
                dgroup.create_dataset('Peaks', rawpeaks.shape, dtype='f', data=rawpeaks,
                                      compression='gzip')
                f[dfile + '/' + s + '/Peaks'].attrs['wtime'] = wtime
                f[dfile + '/' + s + '/Peaks'].attrs['low'] = ifreq
                f[dfile + '/' + s + '/Peaks'].attrs['high'] = ffreq
                f[dfile + '/' + s + '/Peaks'].attrs['threshold'] = threshold

        f.close()
