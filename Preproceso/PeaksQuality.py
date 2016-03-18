"""
.. module:: PeaksQuality

PeaksQuality
*************

:Description: PeaksQuality

    

:Authors: bejar
    

:Version: 

:Created on: 18/03/2016 12:31 

"""
from pylab import *
from sklearn.neighbors import NearestNeighbors
from Config.experiments import experiments
from joblib import Parallel, delayed
from util.plots import show_signal
import argparse
from scipy.optimize import curve_fit
from util.Quality import gauss, bigauss, lorentz, bilorentz, doublelorentz, doublegauss, doublebigauss, triplegauss
import numpy as np

# Plot a set of signals
def plotSignalValues(signals):
    fig = plt.figure()
    minaxis=-0.1
    maxaxis=0.4
    num=len(signals)
    for i in range(num):
        sp1=fig.add_subplot(1,num,i+1)
        sp1.axis([0, signals[0].shape[0],minaxis,maxaxis])
        t = arange(0.0, signals[0].shape[0], 1)
        sp1.plot(t,signals[i])
    plt.show()


def square(x, cf0, cf1, cf2, cf3):
    return ((cf2 * x) **5) + ((cf3 * x) **3) +  cf1 *x + cf0


# fits a function to a vector of data
# data - the data to use to fit the function
# p0 - initial parameters
def fitPeak(func, data, p0):
    peakLength=data.shape[0]
    try:
        coeff, var_matrix = curve_fit(func, np.linspace(0, peakLength-1, peakLength), data)
        valf = func(np.linspace(0, peakLength-1, peakLength), *coeff)
    except RuntimeError:
        valf = np.zeros(peakLength)
    return valf


__author__ = 'bejar'

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()
    lexperiments = args.exp

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511'
        lexperiments = ['e150514']



    for expname in lexperiments:
        datainfo = experiments[expname]

        for sensor in datainfo.sensors:
            print(sensor)

            for dfile in datainfo.datafiles:
                print(dfile)
                f = datainfo.open_experiment_data(mode='r')

                data = datainfo.get_peaks_resample(f, dfile, sensor)
                half = int(data.shape[0]/2.0)
                param = [1.0, data.shape[0], 1 , half]
                for d in data:

                    ffit = fitPeak(gauss, d, param)
                    plotSignalValues([d, ffit])
                datainfo.close_experiment_data(f)

