"""
.. module:: StepsCross

StepsCross
*************

:Description: StepsCross

    

:Authors: bejar
    

:Version: 

:Created on: 25/10/2017 14:09 

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

    dfiles = ['01140515_red.csv',  '02281113_red.csv',  '03060911_red.csv',  '04280213_red.csv',  '05160512_red.csv', '07160611_red.csv']  # '06180912.csv'

    for df1 in dfiles:
        data1 = pd.read_csv(df1, index_col=0)

        for df2 in dfiles:
            if dfiles.index(df2) > dfiles.index(df1):

                data2 = pd.read_csv(df2, index_col=0)

                mdata = pd.merge(data1, data2, left_index=True, right_index=True)
                corr = mdata.corr(method='kendall')

               # print corr
               #  mask = np.zeros_like(corr, dtype=np.bool)
               #  mask[np.triu_indices_from(mask)] = True

                f, ax = plt.subplots(figsize=(11, 9))
                # cmap = sn.diverging_palette(220, 10, as_cmap=True)

                # sn.heatmap(corr, mask=mask, vmax=1, vmin=0, center=0.5, cmap=plt.get_cmap('Reds'), #ListedColormap(sn.color_palette("Reds")),
                #         square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=False)
                sn.heatmap(corr, vmax=1, vmin=0.5, center=0.75, cmap=plt.get_cmap('Reds'), fmt=".2f", annot=False,
                        linewidths=.5, cbar_kws={"shrink": .5})
                plt.yticks(rotation=0)
                plt.xticks(rotation=90)
                plt.title(df1.split('.')[0][2:] + ' ' + df2.split('.')[0][2:])
                plt.savefig('heatcross-' + df1.split('.')[0][2:] + '-' + df2.split('.')[0][2:] + '.pdf', orientation='landscape', format='pdf')
                # plt.show()
                plt.close()
