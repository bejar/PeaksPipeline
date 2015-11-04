"""
.. module:: SFFT

SFFT
*************

:Description: SFFT

    

:Authors: bejar
    

:Version: 

:Created on: 19/06/2015 11:36 

"""

__author__ = 'bejar'


import scipy
import numpy as np

def stft(x, fftsize=1024, overlap=4):
    hop = fftsize / overlap
    w = scipy.hanning(fftsize+1)[:-1]      # better reconstruction with this trick +1)[:-1]
    l = []
    for i in range(0, len(x)-fftsize, hop):
        v = np.fft.rfft(w*x[i:i+fftsize])
        l.append(np.abs(v)**2/np.max(np.abs(v)**2))
    return np.array(l)
