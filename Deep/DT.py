"""
.. module:: DT

DT
*************

:Description: DT

    

:Authors: bejar
    

:Version: 

:Created on: 10/10/2016 8:46 

"""

__author__ = 'bejar'


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


        for dfile, ename in zip(datainfo.datafiles, datainfo.expnames):
            for sensor, nclust in zip(datainfo.sensors, datainfo.clusters):
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
                # scaler = MinMaxScaler(feature_range=(0, 1))
                # dataset = scaler.fit_transform(np.array(lval, dtype=np.float64).reshape(-1, 1))


                logging.info('OPT= %s LSTM DROP W= %f EPOCH= %d LOOKBACK= %d LSTMs= %d NUnits= %d nclasses= %d', opt, drop, epoch, look_back, nLSTM, nunits, nclasses)

                logging.info('%s %s', dfile, sensor)

                dataset = np.array(lval).reshape(-1, 1)
                train_size = len(dataset) - 500 - look_back  # int(len(dataset) * 0.9)
                # test_size = 100 # len(dataset) - train_size
                train, test = dataset[0:train_size,:], dataset[train_size + look_back:len(dataset),:]
                trainX, trainY = create_dataset(train, look_back, classes=nclasses)
                testX, testY = create_dataset(test, look_back, classes=nclasses)

                # model = KNeighborsClassifier(n_neighbors=11, weights='distance')
                # model = DecisionTreeClassifier(max_leaf_nodes=100)
                model = GradientBoostingClassifier(n_estimators=1000, max_depth=5)
                model.fit(trainX, trainY)

                trainPredict = model.predict(trainX)
                testPredict = model.predict(testX)
                acc = 0.0
                for i in range(trainPredict.shape[0]):
                    if trainPredict[i]==trainY[i]:
                        acc += 1.0
                logging.info('Train= %f', acc/trainPredict.shape[0])
                acc = 0.0
                lpred = ""
                ltrain = ""

                confmat = np.zeros((nclasses, nclasses))
                for i in range(testPredict.shape[0]):
                    if testPredict[i]==testY[i]:
                        acc += 1.0
                    confmat[testPredict[i], testY[i]] += 1
                    lpred += voc[testPredict[i]]
                    ltrain += voc[testY[i]]


                logging.info('Test= %f', acc/testPredict.shape[0])
                logging.info('T= %s', ltrain)
                logging.info('P= %s', lpred)


                # for i in range(nclasses):
                #     print voc[i],
                #     for j in range(nclasses):
                #         print int(confmat[i, j]),
                #     print

