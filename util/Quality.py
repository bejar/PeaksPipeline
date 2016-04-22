"""
.. module:: Quality

Quality
*************

:Description: Quality

    

:Authors: bejar
    

:Version: 

:Created on: 18/03/2016 12:29 

"""

__author__ = 'bejar'

from numpy import loadtxt
from numpy.fft import rfft, irfft
import matplotlib.pyplot as plt
from pylab import *
from time import sleep
from scipy.optimize import curve_fit
from scipy.signal import argrelmax
import time
from util.plots import plotListSignals


def lorentz(x, x0, gamma0, h0):
    return h0 + (gamma0 / (((x - x0) * (x - x0)) + (gamma0 * gamma0)))


def doublelorentz(x, x0, gamma0, h0, x1, gamma1, h1):
    return lorentz(x, x0, gamma0, h0) * lorentz(x, x1, gamma1, h1)


def bilorentz(x, x0, gamma1, gamma2, h0):
    v = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] > x0:
            v[i] = lorentz(x[i], x0, gamma1, h0)
        else:
            v[i] = lorentz(x[i], x0, gamma2, h0)
    return (v)


def gauss(x, A, mu, sigma, h0):
    return h0 + (A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)))


def doublegauss(x, A1, mu1, sigma1, h1, A2, mu2, sigma2, h2):
    return gauss(x, A1, mu1, sigma1, h1) * gauss(x, A2, mu2, sigma2, h2)


def triplegauss(x, A1, mu1, sigma1, h1, A2, mu2, sigma2, h2, A3, mu3, sigma3, h3):
    # return np.max(np.array([gauss(x,A1,mu1,sigma1,h1),gauss(x,A2,mu2,sigma2,h2),gauss(x,A3,mu3,sigma3,h3)]),axis=0)
    return gauss(x, A1, mu1, sigma1, h1) * gauss(x, A2, mu2, sigma2, h2) * gauss(x, A3, mu3, sigma3, h3)


def bigauss(x, A, mu, sigma1, sigma2, h0):
    v = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] > mu:
            v[i] = gauss(x[i], A, mu, sigma1, h0)
        else:
            v[i] = gauss(x[i], A, mu, sigma2, h0)
    return (v)


def doublebigauss(x, A1, mu1, sigma11, sigma12, h1, A2, mu2, sigma21, sigma22, h2):
    return bigauss(x, A1, mu1, sigma11, sigma12, h1) * bigauss(x, A2, mu2, sigma21, sigma22, h2)


# fits a function to a vector of data
# data - the data to use to fit the function
# p0 - initial parameters
def fitPeak(func, data, p0):
    peakLength = data.shape[0]
    try:
        coeff, var_matrix = curve_fit(func, np.linspace(0, peakLength - 1, peakLength), data, p0=p0)
        valf = func(np.linspace(0, peakLength - 1, peakLength), *coeff)
    except RuntimeError:
        valf = np.zeros(peakLength)
    return valf


def cuteness(data, percent):
    """
    Computes the quality of a peak as the ratio among the sum of the central percent of the signal and the
    whole signal
    :param data:
    :return:
    """
    #mval = np.min(data)
    data = np.abs(data)

    sumall = np.sum(data)
    part = int(data.shape[0] * percent)
    mid = int(data.shape[0] / 2)
    sumpart = np.sum(data[mid - int(part / 2): mid + int(part / 2)])
    return sumpart / sumall


if __name__ == '__main__':
    pass
