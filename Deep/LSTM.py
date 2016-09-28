"""
.. module:: LTSM

LTSM
*************

:Description: LTSM

    

:Authors: bejar
    

:Version: 

:Created on: 19/09/2016 13:01 

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
from joblib import Parallel, delayed
from util.plots import show_signal
import argparse
__author__ = 'bejar'


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

voc = '#ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*+-%&/<>[]{}()!?#'

if __name__ == '__main__':

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

        for dfile, ename in zip(datainfo.datafiles, datainfo.expnames):
            for sensor, nclust in zip(datainfo.sensors, datainfo.clusters):
                rfile = open(datainfo.dpath + '/'+ datainfo.name + '/Results/stringseq-' + dfile+'1' + '-' + ename + '-' + sensor + '.txt', 'r')
                seq = ""
                for line in rfile:
                    seq += line.strip()
                rfile.close()
                lval = [voc.index(v) for v in seq]
                scaler = MinMaxScaler(feature_range=(0, 1))
                dataset = scaler.fit_transform(np.array(lval, dtype=np.float64).reshape(-1, 1))
                #dataset = np.array(lval).reshape(-1, 1)
                train_size = int(len(dataset) * 0.67)
                test_size = len(dataset) - train_size
                train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

                look_back = 2
                trainX, trainY = create_dataset(train, look_back)
                testX, testY = create_dataset(test, look_back)
                trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
                testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
                model = Sequential()
                # model.add(LSTM(10, input_dim=look_back, return_sequences=True))
                model.add(LSTM(100, input_dim=look_back))
                model.add(Dense(1, activation='linear'))
                model.compile(loss='mean_squared_error', optimizer='rmsprop')
                model.fit(trainX, trainY, nb_epoch=50, batch_size=1, verbose=2)

                trainScore = model.evaluate(trainX, trainY, verbose=0)
                trainScore = math.sqrt(trainScore)
                trainScore = scaler.inverse_transform(np.array([[trainScore]]))
                print('Train Score: %.2f RMSE' % (trainScore))
                testScore = model.evaluate(testX, testY, verbose=0)
                testScore = math.sqrt(testScore)
                testScore = scaler.inverse_transform(np.array([[testScore]]))
                print('Test Score: %.2f RMSE' % (testScore))

                trainPredict = model.predict(trainX)
                testPredict = model.predict(testX)

                # shift train predictions for plotting
                trainPredictPlot = np.empty_like(dataset)
                trainPredictPlot[:, :] = np.nan
                trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

                # shift test predictions for plotting
                testPredictPlot = np.empty_like(dataset)
                testPredictPlot[:, :] = np.nan
                testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

                # plot baseline and predictions
                plt.plot(dataset)
                plt.plot(trainPredictPlot)
                plt.plot(testPredictPlot)
                plt.show()
                plt.close()