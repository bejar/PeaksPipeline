"""
.. module:: DTTest

DTTest
*************

:Description: DTTest

    

:Authors: bejar
    

:Version: 

:Created on: 13/10/2016 8:28 

"""

import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from pylab import *
from sklearn.neighbors import NearestNeighbors
from Config.experiments import experiments
#from joblib import Parallel, delayed
from util.plots import show_signal
import argparse
import logging
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

def create_dataset(dataset, look_back=1, classes=13):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

voc = '#ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*+-%&/<>[]{}()!?#'

if __name__ == '__main__':

    ## Console Logging


    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()
    lexperiments = args.exp

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511'
        lexperiments = ['e150514alt']


    for expname in lexperiments:
        datainfo = experiments[expname]
        opt = 'adam'
        drop = 0.02
        epoch = 200
        nLSTM = 2
        nunits = 100
        look_back = 150
        nclasses = 13

        # ---- Logging
        now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        logging.basicConfig(filename=datainfo.dpath + '/' + datainfo.name + '/Results/' + datainfo.name + '-LSTM-' + now + '.txt', filemode='w',
                            level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')


        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
        # ---------------------

        dfile = datainfo.datafiles[0]
        ename = datainfo.expnames[0]
        sensor = datainfo.sensors[0]

        # for dfile, ename in zip(datainfo.datafiles, datainfo.expnames):
        #     for sensor, nclust in zip(datainfo.sensors, datainfo.clusters):
        for ntrees in range(100, 501, 100):
            for tsize in range(5, 26, 5):
                for look_back in [5, 10, 25, 50]:

                    rfile = open(datainfo.dpath + '/'+ datainfo.name + '/Results/stringseq-' + dfile + '1' + '-' + ename + '-' + sensor + '.txt', 'r')
                    seq = ""
                    for line in rfile:
                        seq += line.strip()
                    rfile.close()
                    lval = []
                    prev = None
                    for v in seq:
                        if v == '#' and prev == '#':
                            pass
                        else:
                            lval.append(voc.index(v))
                        prev = v

                    dataset = np.array(lval).reshape(-1, 1)
                    train_size = len(dataset) - 300 - look_back  # int(len(dataset) * 0.9)
                    # test_size = 100 # len(dataset) - train_size
                    test = dataset[train_size + look_back:len(dataset),:]
                    testX, testY = create_dataset(test, look_back, classes=nclasses)

                    part = 5
                    for p in range(part):
                        train = dataset[p*int(train_size/part):(p+1) * int(train_size/part),:]
                        trainX, trainY = create_dataset(train, look_back, classes=nclasses)
                        model = GradientBoostingClassifier(n_estimators=ntrees, max_depth=tsize)
                        model.fit(trainX, trainY)

                        trainPredict = model.predict(trainX)
                        testPredict = model.predict(testX)
                        acc = 0.0
                        for i in range(trainPredict.shape[0]):
                            if trainPredict[i]==trainY[i]:
                                acc += 1.0

                        acct = 0.0
                        lpred = ""
                        ltrain = ""

                        confmat = np.zeros((nclasses, nclasses))
                        for i in range(testPredict.shape[0]):
                            if testPredict[i]==testY[i]:
                                acct += 1.0
                            confmat[testPredict[i], testY[i]] += 1
                            lpred += voc[testPredict[i]]
                            ltrain += voc[testY[i]]

                        logging.info('%d %s %s NTREES= %d TSIZE= %d LOOKBACK= %d Train= %f Test= %f',
                                     p, dfile, sensor, ntrees, tsize, look_back,
                                     acc/trainPredict.shape[0], acct/testPredict.shape[0])
