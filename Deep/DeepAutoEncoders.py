"""
.. module:: DeepAutoEncoders

DeepAutoEncoders
******

:Description: DeepAutoEncoders

    Different Auxiliary functions used for different purposes

:Authors:
    bejar

:Version: 

:Date:  24/03/2016
"""
from __future__ import print_function

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from SdA import SdA
from Config.experiments import experiments
from util.Quality import cuteness
from util.plots import plotListSignals


__author__ = 'bejar'

if __name__ == '__main__':


    learning_rate=0.5
    training_epochs=20000
    batch_size=300
    pretraining_epochs=500
    pretrain_lr=0.5
    finetune_lr=0.1

    datainfo = experiments['e150514']
    dfile = datainfo.datafiles[0]
    sensor = datainfo.sensors[2]
    f = datainfo.open_experiment_data(mode='r+')
    data = datainfo.get_peaks_resample(f, dfile, sensor)
    datainfo.close_experiment_data(f)
    cute_data = []
    for d in data:
        if cuteness(d, 0.3) >0.5:
            cute_data.append(d)

    lcd = len(cute_data)
    train_set_x = theano.shared(numpy.asarray(numpy.vstack(cute_data),
                                               dtype=theano.config.floatX),
                                 borrow=True)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size

    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print('... building the model')
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=data.shape[1],
        hidden_layers_sizes=[1000, 500, 20, data.shape[1]],
        n_outs=10
    )
    # end-snippet-3 start-snippet-4

    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)
    print('... pre-training the model')
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    corruption_levels = [.1, .1, .1, .1]
    for i in range(sda.n_layers):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, numpy.mean(c)))

    end_time = timeit.default_timer()

    print(sda.dA_layers[-1].W.get_value(borrow=True).shape)
    plotListSignals(sda.dA_layers[-1].W.get_value(borrow=True), ncols=5)