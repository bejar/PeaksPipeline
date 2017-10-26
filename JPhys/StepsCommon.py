"""
.. module:: StepsCross

StepsCross
*************

:Description: StepsCross

    

:Authors: bejar
    

:Version: 

:Created on: 25/10/2017 13:22 

"""
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from matplotlib.colors import ListedColormap

__author__ = 'bejar'

if __name__ == '__main__':
    dfiles = ['01140515.csv',  '02281113.csv',  '03060911.csv',  '04280213.csv',  '05160512.csv', '07160611.csv']  # '06180912.csv'
    sensors = ['L4ci', 'L4cd', 'L5ri', 'L5rd', 'L5cd', 'L5ci', 'L6ri', 'L6rd', 'L6ci', 'L6cd', 'L7ri', 'L7rd']

    out = set(['L4ci', 'L4cd', 'L7ri', 'L7rd'])

    dexpsensors = {}

    ldm = []

    for df in dfiles:
        data = pd.read_csv(df, sep='\t', index_col=0)
        lindex = []
        for exps in data.index.values:
            if exps.split()[0] not in out and exps.split()[1] not in out:
                lindex.append(exps)

        ndf = data.loc[lindex]
        ndf.to_csv(df.split('.')[0] + '_red.csv')



