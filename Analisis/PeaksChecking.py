"""
.. module:: PeaksChecking

PeaksChecking
******

:Description: PeaksChecking

    Different Auxiliary functions used for different purposes

:Authors:
    bejar

:Version: 

:Date:  23/03/2016
"""

import numpy as np
from scipy.signal import butter, filtfilt

from Config.experiments import experiments, lexperiments
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
from util.plots import plotListSignals

__author__ = 'bejar'

# ---------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")


    args = parser.parse_args()
    lexperiments = args.exp

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511'
        lexperiments = ['e150514']

    for exp in lexperiments:

        datainfo = experiments[exp]

        for sensor in datainfo.sensors:
            print(sensor)
            for dfile in [datainfo.datafiles[0]]:
                print(dfile)

                f = datainfo.open_experiment_data(mode='r')
                data = datainfo.get_peaks_resample_PCA(f, dfile, sensor)

                for d in data:
                    mn = np.mean(d[0:20])
                    print d[0:20]
                    plotListSignals([d])

                datainfo.close_experiment_data(f)


