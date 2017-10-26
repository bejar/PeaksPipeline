"""
.. module:: StepCorrelation

StepCorrelation
*************

:Description: StepCorrelation

    

:Authors: bejar
    

:Version: 

:Created on: 18/10/2017 11:49 

"""

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from matplotlib.colors import ListedColormap
from matplotlib import cm

__author__ = 'bejar'


if __name__ == '__main__':
    dfiles = ['01140515.csv',  '02281113.csv',  '03060911.csv',  '04280213.csv',  '05160512.csv',  '06180912.csv',  '07160611.csv']

    for fl in dfiles:
        data = pd.read_csv(fl, sep='\t', index_col=0)
        print data.columns

        corr = data.corr(method='kendall') # spearman kendall

        # print corr
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(11, 9))
        # cmap = sn.diverging_palette(220, 10, as_cmap=True)

        sn.heatmap(corr, mask=mask, vmax=1, vmin=0, center=0.5, cmap=plt.get_cmap('Reds'), #ListedColormap(sn.color_palette("Reds")),
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=False)
        plt.yticks(rotation=0)
        plt.title(fl.split('.')[0][2:])
        plt.savefig('corr-' + fl.split('.')[0][2:] + '.pdf', orientation='landscape', format='pdf')
        # plt.show()
        plt.close()