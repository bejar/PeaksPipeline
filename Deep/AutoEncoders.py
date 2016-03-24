"""
.. module:: AutoEncoders

AutoEncoders
******

:Description: AutoEncoders

    Different Auxiliary functions used for different purposes

:Authors:
    bejar

:Version: 

:Date:  23/03/2016
"""
from __future__ import print_function

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from dA import dA
from Config.experiments import experiments
from util.Quality import cuteness
from util.plots import plotListSignals

__author__ = 'bejar'


if __name__ == '__main__':


    learning_rate=0.5
    training_epochs=20000
    batch_size=300

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
    print(lcd)



    train_set_x = theano.shared(numpy.asarray(numpy.vstack(cute_data),
                                               dtype=theano.config.floatX),
                                 borrow=True)
    # train_set_y =theano.shared( numpy.asarray(numpy.vstack(cute_data[(lcd//2)+1:]),
    #                                            dtype=theano.config.floatX),
    #                              borrow=True)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # start-snippet-2
    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')
    # end-snippet-2


    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible= data.shape[1],
        n_hidden=20
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()


    # go through training epochs
    for epoch in range(training_epochs):
        # go through trainng set
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        print('Training epoch %d, cost ' % epoch, numpy.mean(c))

    end_time = timeit.default_timer()

    print(da.W.get_value(borrow=True).shape)
    plotListSignals(da.W.get_value(borrow=True).T, ncols=5)