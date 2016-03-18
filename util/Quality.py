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

##
## Identifies a set of peaks from the cat signals
##

# Plot a set of signals
def plotSignalValues(signals):
    fig = plt.figure()
    minaxis=-0.1
    maxaxis=0.4
    num=len(signals)
    for i in range(num):
        sp1=fig.add_subplot(1,num,i+1)
        sp1.axis([0,peakLength,minaxis,maxaxis])
        t = arange(0.0, peakLength, 1)
        sp1.plot(t,signals[i])
    plt.show()


def lorentz(x,x0,gamma0,h0):
    return h0+(gamma0/(((x-x0)*(x-x0))+(gamma0*gamma0)))

def doublelorentz(x,x0,gamma0,h0,x1,gamma1,h1):
    return lorentz(x,x0,gamma0,h0)*lorentz(x,x1,gamma1,h1)

def bilorentz(x, x0, gamma1,gamma2,h0):
    v=np.zeros(len(x))
    for i in range(len(x)):
        if x[i]>x0:
            v[i]=lorentz(x[i],x0,gamma1,h0)
        else:
            v[i]=lorentz(x[i],x0,gamma2,h0)
    return(v)

def gauss(x, A, mu, sigma, h0):
    return h0 + (A*np.exp(-(x-mu)**2/(2.*sigma**2)))

def doublegauss(x, A1, mu1, sigma1, h1, A2, mu2, sigma2, h2):
    return gauss(x,A1,mu1,sigma1,h1)*gauss(x,A2,mu2,sigma2,h2)

def triplegauss(x, A1, mu1, sigma1,h1,A2, mu2, sigma2,h2,A3, mu3, sigma3,h3):
    #return np.max(np.array([gauss(x,A1,mu1,sigma1,h1),gauss(x,A2,mu2,sigma2,h2),gauss(x,A3,mu3,sigma3,h3)]),axis=0)
    return gauss(x,A1,mu1,sigma1,h1)*gauss(x,A2,mu2,sigma2,h2)*gauss(x,A3,mu3,sigma3,h3)

def bigauss(x, A, mu, sigma1,sigma2,h0):
    v=np.zeros(len(x))
    for i in range(len(x)):
        if x[i]>mu:
            v[i]=gauss(x[i],A,mu,sigma1,h0)
        else:
            v[i]=gauss(x[i],A,mu,sigma2,h0)
    return(v)

def doublebigauss(x, A1, mu1, sigma11, sigma12, h1, A2, mu2, sigma21, sigma22, h2):
    return bigauss(x,A1,mu1,sigma11,sigma12,h1)*bigauss(x,A2,mu2,sigma21,sigma22,h2)


# fits a function to a vector of data
# data - the data to use to fit the function
# p0 - initial parameters
def fitPeak(func, data, p0):
    peakLength=data.shape[0]
    try:
        coeff, var_matrix = curve_fit(func, np.linspace(0,peakLength-1,peakLength), data,p0=p0)
        valf=func(np.linspace(0,peakLength-1,peakLength),*coeff)
    except RuntimeError:
        valf=np.zeros(peakLength)
    return valf

###
# Finds peaks in a vector
# data - vector of data
# peakLength - Length of the window used to identify the peak
# fftLenthg - Length of the window used to perform the fft
# centertol - tolearnce of the position of the maximum to be considered in the center
# freq - number of frequencies to keep from the fft for denoising
# minHehight - minimum height of the data in the window
def findPeaks(data,peakLength,fftLength, centertol, freq,minHeight):
    offset=int((fftLength-peakLength)/2.0)
    halffftLength=int(fftLength/2.0)
    halfpeakLength=int(peakLength/2.0)
    peakCount=0
    p=0
    advance=50

    lfitfun=[(gauss,[1.0,halfpeakLength,halfpeakLength/2.0,0.0]),
             (doublegauss, [1.0, halfpeakLength - halfpeakLength / 2.0, halfpeakLength / 3.0, 0.0, 1.0, halfpeakLength + halfpeakLength / 2.0, halfpeakLength / 3.0, 0.0]),
             (triplegauss,[1.0,halfpeakLength-halfpeakLength/2.0,halfpeakLength/3.0,0.0,1.0,halfpeakLength,halfpeakLength/3.0,0.0,1.0,halfpeakLength+halfpeakLength/2.0,halfpeakLength/3.0,0.0] ),
             (bigauss,[1.0,halfpeakLength,halfpeakLength/2.0,halfpeakLength/2.0,0.0] )
            ]

    while p+fftLength<data.shape[0]:
        if np.max(data[p:fftLength+p])>minHeight:
            # copy the current segment
            orig=data[p:fftLength+p]
            # Apply the FFT and denoise
            temp= rfft(orig)
            temp[freq:len(temp)]=0
            vals= irfft(temp)

            # Fits the position of the maximum
            # Correct the position with the offset
            maxp=np.argmax(vals[offset:offset+peakLength])+offset
            # If the maximum is in the middle of the window with a tolerance
            # keep the peak
            if maxp >= halffftLength-centertol and maxp <= halffftLength+centertol:
         #       plotSignalValues(sp,[vals[offset:offset+peakLength]])
                print 'Peak Centered Position: ',p+maxp

                lfitsig=[orig[offset:offset+peakLength],vals[offset:offset+peakLength]]

                for fun,p0 in lfitfun:
                    resfit=fitPeak(fun,orig[offset:offset+peakLength],p0)
                    print argrelmax(resfit)
                    lfitsig.append(resfit)
                    ssq= orig[offset:offset+peakLength] - resfit
                    print np.sum(ssq *ssq)
                plotSignalValues(lfitsig)

                # next peak is after the length of the detected peak
                p=p+offset+advance#+peakLength
                peakCount+=1
            # If the maximum is not centered move so it is in the next window
            else:
                if maxp>halffftLength:
                    p=p+(maxp -halffftLength)
                else:
                    p+=maxp
                print 'Peak not centered, jump to: ',p, maxp
        #    sleep(2)
        #    plt.close()
        #    fig = plt.figure()
        else:
            p+=halffftLength
    return peakCount

if __name__ == '__main__':

    cpath='/home/bejar/Dropbox/Gennaro/Gatos/Lab Rudomin/FILTRO GE/data/'

    data = loadtxt(cpath+'CTRL300103.csv', skiprows=1, usecols=[1])


    peakLength=300 # Length of the peak
    minHeight=0.05 # Minimum height of the peak
    fftLength=512
    freq=20# Number of frequencies of the FFT
    centertol=10 # Tolerance of peak centering

    #plt.ion()
    #plt.hold(True)
    t0 = time.time()

    peakCount=findPeaks(data,peakLength,fftLength, centertol, freq,minHeight)
    totalTime = time.time() - t0

    print peakCount, totalTime

