"""
.. module:: ViewPeakSignal

ViewPeakSignal
*************

:Description: ViewPeakSignal

    

:Authors: bejar
    

:Version: 

:Created on: 09/11/2016 14:01 

"""

from util.plots import show_signal, plotSignals, show_two_signals, show_list_signals
import argparse

from Config.experiments import experiments

__author__ = 'bejar'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")
    parser.add_argument('--extra', help="Procesa sensores extra del experimento", action='store_true', default=False)
    parser.add_argument('--raw', nargs='+', default=[], help="Procesado del pico")

    args = parser.parse_args()
    lexperiments = args.exp

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511'e160317
        lexperiments = ['e150514']
        args.extra = True
        args.raw = 1


    expname = lexperiments[0]

    datainfo = experiments[expname]
    print(datainfo.dpath + datainfo.name + '/' + datainfo.name)
    f = datainfo.open_experiment_data(mode='r')


    if not args.extra:
        lsensors = datainfo.sensors
    else:
        lsensors = datainfo.extrasensors


    for sensor in [lsensors[0]]:
        print(sensor)

        for dfile in [datainfo.datafiles[0]]:
            if args.raw == 0:
                data = datainfo.get_peaks(f, dfile, sensor)
            elif args.raw == 1:
                data = datainfo.get_peaks_resample(f, dfile, sensor)
            else:
                data = datainfo.get_peaks_resample_PCA(f, dfile, sensor)

            for i in range(data.shape[0]):
                if not args.extra:
                    show_signal(data[i])
                else:
                    show_signal(-data[i])