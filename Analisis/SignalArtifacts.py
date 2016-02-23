"""
.. module:: SignalArtifacts

SignalArtifacts
*************

:Description: SignalArtifacts

    Busca artefactos en la senyal raw

:Authors: bejar
    

:Version: 

:Created on: 22/02/2016 12:58 

"""

import numpy as np
import argparse
from pylab import *


from Config.experiments import experiments
__author__ = 'bejar'


def show_signal(signal, title='', mid=0, top=0, bottom=0):
    """
    Plots a signal
    :param signal:
    :return:
    """
    matplotlib.rcParams.update({'font.size': 26})
    fig = plt.figure()
    fig.set_figwidth(30)
    fig.set_figheight(40)
    fig.suptitle(title, fontsize=48)
    minaxis = min(min(signal), bottom)
    maxaxis = max(max(signal), top)
    num = len(signal)
    sp1 = fig.add_subplot(111)
    sp1.axis([0, num, minaxis, maxaxis])
    t = arange(0.0, num, 1)
    sp1.plot(t, signal)
    plt.axhline(linewidth=4, color='b', y=top)
    plt.axhline(linewidth=4, color='b', y=bottom)
    plt.axhline(linewidth=4, color='r', y=mid)

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    chunk = 100000
    delta = 500

    args = parser.parse_args()
    lexperiments = args.exp

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511''e140225''e130221' 'e130221'
        lexperiments = ['e150707']

    for expname in [lexperiments[0]]:

        datainfo = experiments[expname]

        f = datainfo.open_experiment_data(mode='r+')


        for dfile in datainfo.datafiles:

            rdata = datainfo.get_raw_data(f, dfile)

            for i, s in enumerate(datainfo.sensors):
                dmn = np.mean(rdata[0:chunk, i])
                dst = np.std(rdata[0:chunk, i])
                # print dmn, dst
                lout = [0]
                for j in range(rdata.shape[0]):
                    if not (((dmn + (6* dst)) > rdata[j, i] > (dmn - (6* dst)))) and (j > (lout[-1] + delta)):
                        # print rdata[j, i], j
                        lout.append(j)
                        show_signal(rdata[j-delta:j+delta, i], mid=dmn, top=dmn + (6* dst), bottom=dmn - (6* dst) )
                print s, len(lout)-1


