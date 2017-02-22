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
from keras.callbacks import RemoteMonitor
from sklearn.preprocessing import MinMaxScaler
from pylab import *
from sklearn.neighbors import NearestNeighbors
from Config.experiments import experiments
#from joblib import Parallel, delayed
from util.plots import show_signal
import argparse
import logging
import time

__author__ = 'bejar'


from keras.callbacks import Callback
import json

class MyRemoteMonitor(Callback):
    '''Callback used to stream events to a server.
    Requires the `requests` library.
    # Arguments
        root: root url to which the events will be sent (at the end
            of every epoch). Events are sent to
            `root + '/publish/epoch/end/'` by default. Calls are
            HTTP POST, with a `data` argument which is a
            JSON-encoded dictionary of event data.
    '''

    def __init__(self,
                 id = '',
                 root='http://localhost:9000',
                 path='/publish/epoch/end/',
                 field='data',
                 headers={'Accept': 'application/json', 'Content-Type': 'application/json'}):
        super(Callback, self).__init__()
        self.id = id
        self.root = root
        self.path = path
        self.field = field
        self.headers = headers

    def on_epoch_end(self, epoch, logs={}):
        import requests
        send = {}
        send['epoch'] = epoch
        for k, v in logs.items():
            send[k] = v
        try:
            requests.post(self.root + self.path,
                          {self.field: json.dumps(send), 'id': self.id},
                          headers=self.headers)
        except:
            print('Warning: could not reach RemoteMonitor '
                  'root server at ' + str(self.root))

def create_dataset(dataset, look_back=1, classes=13):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        clvector = np.zeros(classes)
        clvector[dataset[i + look_back, 0]] = 1
        dataY.append(clvector)
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
        epoch = 300
        nLSTM = 2
        nunits = 100
        look_back = 20
        nclasses = 12

        # ---- Logging
        now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        logging.basicConfig(filename=datainfo.dpath + '/' + datainfo.name + '/Results/' + datainfo.name + '-LSTM-' + now + '.txt', filemode='w',
                            level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')

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
                lval = [voc.index(v)-1 for v in seq if v != '#']
                # scaler = MinMaxScaler(feature_range=(0, 1))
                # dataset = scaler.fit_transform(np.array(lval, dtype=np.float64).reshape(-1, 1))


                logging.info('OPT= %s LSTM DROP W= %f EPOCH= %d LOOKBACK= %d LSTMs= %d NUnits= %d nclasses=%d', opt, drop, epoch, look_back, nLSTM, nunits, nclasses)

                logging.info('%s %s', dfile, sensor)

                dataset = np.array(lval).reshape(-1, 1)
                train_size = len(dataset) - 100 - look_back  # int(len(dataset) * 0.9)
                # test_size = 100 # len(dataset) - train_size
                train, test = dataset[0:train_size,:], dataset[train_size + look_back:len(dataset),:]
                trainX, trainY = create_dataset(train, look_back, classes=nclasses)
                testX, testY = create_dataset(test, look_back, classes=nclasses)
                trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
                testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
                model = Sequential()
                #model.add(LSTM(100, input_dim=look_back, return_sequences=True, dropout_W=drop, activation='tanh'))
                for i in range(1, nLSTM):
                    model.add(LSTM(nunits, input_dim=look_back, return_sequences=True, dropout_W=drop, activation='tanh'))
                model.add(LSTM(nunits, input_dim=look_back, dropout_W=drop, activation='tanh'))
                #model.add(Dense(50, activation='relu'))
                model.add(Dense(nclasses, activation='softmax'))
                model.compile(loss='categorical_crossentropy', optimizer=opt)

                tboard = MyRemoteMonitor(id='v', root='http://localhost:8850', path='/Update')
                model.fit(trainX, trainY, nb_epoch=epoch, batch_size=1, verbose=2, callbacks=[tboard])

                # trainScore = model.evaluate(trainX, trainY, verbose=0)
                # trainScore = math.sqrt(trainScore)
                # #trainScore = scaler.inverse_transform(np.array([[trainScore]]))
                # print('Train Score: %.2f RMSE' % (trainScore))
                # testScore = model.evaluate(testX, testY, verbose=0)
                # testScore = math.sqrt(testScore)
                # #testScore = scaler.inverse_transform(np.array([[testScore]]))
                # print('Test Score: %.2f RMSE' % (testScore))

                trainPredict = model.predict(trainX)
                testPredict = model.predict(testX)
                acc = 0.0
                for i in range(trainPredict.shape[0]):
                    if np.argmax(trainPredict[i]) == np.argmax(trainY[i]):
                        acc += 1.0
                logging.info('Train= %f', acc/trainPredict.shape[0])
                acc = 0.0
                lpred = ""
                ltrain = ""
                confmat = np.zeros((nclasses, nclasses))
                for i in range(testPredict.shape[0]):
                    if np.argmax(testPredict[i]) == np.argmax(testY[i]):
                        acc += 1.0
                    confmat[np.argmax(testPredict[i]), np.argmax(testY[i])] += 1
                    lpred += voc[np.argmax(testPredict[i])]
                    ltrain += voc[np.argmax(testY[i])]

                logging.info('Test= %f', acc/testPredict.shape[0])
                logging.info('T= %s', ltrain)
                logging.info('P= %s', lpred)

                for i in range(nclasses):
                    print voc[i],
                    for j in range(nclasses):
                        print int(confmat[i, j]),
                    print

                # # shift train predictions for plotting
                # trainPredictPlot = np.zeros((len(dataset), 12))
                # trainPredictPlot[:, :] = np.nan
                # trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
                #
                # # shift test predictions for plotting
                # testPredictPlot = np.empty_like(dataset)
                # testPredictPlot[:, :] = np.nan
                # testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
                #
                # # plot baseline and predictions
                # plt.plot(dataset)
                # plt.plot(trainPredictPlot)
                # plt.plot(testPredictPlot)
                # plt.show()
                # plt.close()
