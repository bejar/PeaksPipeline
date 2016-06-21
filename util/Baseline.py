"""
.. module:: Baseline

Baseline
*************

:Description: Baseline

    

:Authors: bejar
    

:Version: 

:Created on: 17/06/2016 10:02 

"""

__author__ = 'bejar'

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def baseline_als(y, lam, p, niter=10):
    """
    Asymmetric Least Squares Smoothing

    Method for smoothing also useful for baseline correction

    Taken from

    @article{eilers2005baseline,
     title={Baseline correction with asymmetric least squares smoothing},
     author={Eilers, Paul HC and Boelens, Hans FM},
     journal={Leiden University Medical Centre Report},
     year={2005}
    }

    :param y: signal
    :param lam: signal smoothing, usual values 10^2 - 10^9
    :param p: asymmetry usual values from 0.001 to 0.1 for baseline removal
                 (but for smoothing can be close to 0.9)
    :param niter: number of iterations,
    :return:
    """
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in xrange(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


def find_baseline(x, resolution=50):
    """
    approximates the baseline of a peak searching for the value that returns more crossings to a flat line

    :param x:
    :return:
    """
    mnval = np.min(x)
    mxval = np.max(x)

    step = (mxval -mnval)/resolution
    mcount = 0
    mcountval = -1
    for v in np.linspace(mnval, mxval, num=resolution):
        nint = np.sum(np.logical_and(x>=v, x<v+step))
        if mcount < nint:
            mcount = nint
            mcountval =  np.sum(x[np.logical_and(x>=v, x<v+step)])/nint
    return mcountval

